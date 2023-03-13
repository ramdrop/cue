# -*- coding: future_fstrings -*-
# public
from logging import exception
import os
import os.path as osp
from os.path import join
import sys
from datetime import datetime
import gc
import numpy as np
import json
from tqdm import tqdm
from matplotlib import pyplot as plt
import shutil

import torch
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig

# private
import matin
import models
import losses

from utlis import files
from utlis import misc
from utlis import pointcloud
from utlis import metrics
from utlis import transform_estimation
from utlis import timer
from utlis import registration
from utlis import knn
from utlis import uncertainty_util

np.set_printoptions(precision=3)

class AlignmentTrainer:

    def __init__(self, config, train_dataloader, val_dataloader=None):
        torch.autograd.set_detect_anomaly(False)

        self.nan_loss_count = 0
        self.nan_grad_count = 0

        num_feats = 1  # occupancy only for 3D Match dataset. For ScanNet, use RGB 3 channels.

        self.checkpoint_dir = os.path.relpath(HydraConfig.get().runtime.output_dir)
        # self.checkpoint_dir = join(config.out_dir, '{}_{}'.format(''.join([x for x in config.trainer if x.isupper()][:-1]), datetime.now().strftime('%m%d_%H%M%S')))
        files.ensure_dir(self.checkpoint_dir)
        files.ensure_dir(join(self.checkpoint_dir, 'models'))
        self.log = matin.ln(__name__, tofile=join(self.checkpoint_dir, 'trainer.log')).get_logger()

        # --------------------------- deterministic -------------------------- #
        # self.golden_seed = 919
        # torch.manual_seed(self.golden_seed)
        # np.random.seed(self.golden_seed)

        # ---------------------------- set device ---------------------------- #
        # torch.cuda.set_device(hero.schedule_device())
        torch.cuda.set_device(0)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log.info(f'device:{self.device} {torch.cuda.current_device()}, log dir: {self.checkpoint_dir}')


        Model = models.load_model(config.model)            # model initialization: 32, bn_momentum=0.05, normalize_feature=true, conv1_kernel_size=5, D=3
        model = Model(num_feats, config.model_n_out, bn_momentum=config.bn_momentum, normalize_feature=config.normalize_feature, conv1_kernel_size=config.conv1_kernel_size, D=3)
        # solver = models.solver_local.StochasticSolver(model, config)

        # snapshot
        snap_dir = join(self.checkpoint_dir, 'models')
        files.ensure_dir(snap_dir)
        shutil.copy(__file__, snap_dir)
        shutil.copy(sys.modules[Model.__module__].__file__, snap_dir)
        shutil.copy('losses/btl.py', snap_dir)

        num_of_para = 0
        for module in model.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                para = 1
                for k in module.kernel.shape:
                    para *= k
                num_of_para += para
        self.log.info(f"model parameters: {num_of_para/1e6:.2f}M")

        # ------------------------- apply MC Dropout ------------------------- #
        if config.mc_p != 0:
            def dropout_hook_wrapper(module, sinput, soutput):
                input = soutput.F
                output = F.dropout(input, p=config.mc_p, training=module.training)
                soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
                return soutput_new
            for module in model.modules():
                if isinstance(module, ME.MinkowskiConvolution):
                    module.register_forward_hook(dropout_hook_wrapper)

        # ------------- freeze mu branch and update sigma branch ------------- #
        if config.phase == 'train_sigma':
            assert config.weights != '', 'undefined mu_model weights'
            checkpoint = torch.load(config.weights)
            self.log.info(f"strictly loading weight from epoch {checkpoint['epoch']}")
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                self.log.info(f'strictly loading weights failed, set strict=False')
                model.load_state_dict(checkpoint['state_dict'], strict=False)

            for name, param in model.named_parameters():    # only train the sigma branch
                # if 'sigma_conv1_tr' in name or 'sigma_final' in name:
                if 'sigma' in name:
                    self.log.info(f'{name}.requires_grad = True')
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if config.phase == 'finetune':
            assert config.weights != '', 'undefined mu_model weights'
            checkpoint = torch.load(config.weights)
            self.log.info(f"loading weight from epoch {checkpoint['epoch']}")
            self.model_aux = Model(num_feats, config.model_n_out, bn_momentum=config.bn_momentum, normalize_feature=config.normalize_feature, conv1_kernel_size=config.conv1_kernel_size, D=3)
            self.model_aux.load_state_dict(checkpoint['state_dict'])
            self.model_aux = self.model_aux.to(self.device)

        self.config = config
        self.model = model
        # self.solver = solver
        self.max_epoch = config.max_epoch
        self.val_max_iter = config.val_max_iter
        self.val_every_epoch = config.val_every_epoch
        self.matching_search_voxel_size = config.voxel_size * config.positive_pair_search_voxel_size_multiplier
        self.best_val_metric = config.best_val_metric
        self.best_val_epoch = -np.inf
        self.best_val = -np.inf
        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh
        self.neg_weight = config.neg_weight
        self.bayesian_margin = config.bayesian_margin

        if config.trainer == 'BTLS':
            self.bayesian_sampling_margin = config.bayesian_sampling_margin
            self.nb_samples = config.nb_samples

        # ------------------------------- loss ------------------------------- #
        # self.bayesian_contrastive_loss = losses.BayesianContrastiveLoss(varPrior=1 / 32.0, mp=self.pos_thresh, mn=self.neg_thresh)
        self.bayesian_triplet_loss = losses.BayesianTripletLoss(varPrior=1 / 32.0, margin=self.bayesian_margin)
        if config.trainer == 'BTLS':
            self.bayesian_triplet_loss_sampling = losses.BayesianTripletLossSampling(margin=self.bayesian_sampling_margin, nb_samples=self.nb_samples)

        # ----------------------------- optimizer ---------------------------- #
        # self.optimizer = getattr(optim, config.optimizer)(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        self.optimizer = getattr(optim, config.optimizer)(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
        self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, config.exp_gamma)

        self.iter_size = config.iter_size
        self.batch_size = train_dataloader.batch_size
        self.data_loader = train_dataloader
        self.val_data_loader = val_dataloader
        self.test_valid = config.test_valid
        self.log_step = int(np.sqrt(self.config.batch_size))
        self.model = self.model.to(self.device)
        # self.solver = self.solver.to(self.device)

        self.start_epoch = 1
        self.curr_iter = 0
        self.writer = SummaryWriter(logdir=self.checkpoint_dir)

        if config.resume is not None:
            if osp.isfile(config.resume):
                self.log.info("=> loading checkpoint '{}'".format(config.resume))
                state = torch.load(config.resume)
                self.start_epoch = state['epoch']
                model.load_state_dict(state['state_dict'])
                self.scheduler.load_state_dict(state['scheduler'])
                self.optimizer.load_state_dict(state['optimizer'])

                if 'best_val' in state.keys():
                    self.best_val = state['best_val']
                    self.best_val_epoch = state['best_val_epoch']
                    self.best_val_metric = state['best_val_metric']
            else:
                raise ValueError(f"=> no checkpoint found at '{config.resume}'")

    def train(self):
        """
        Full training logic
        """
        # ------------------ Baseline random feature performance ----------------- #
        if self.test_valid:
            with torch.no_grad():
                val_dict = self._valid_epoch(0, [0])

            for k, v in val_dict.items():
                self.writer.add_scalar(f'val/{k}', v, 0)

        for epoch in range(self.start_epoch, self.max_epoch + 1):
            lr = self.scheduler.get_last_lr()
            self._train_epoch(epoch)
            self._save_checkpoint(epoch)
            self.scheduler.step()

            if epoch % self.val_every_epoch == 0:
                with torch.no_grad():
                    val_dict = self._valid_epoch(epoch, lr)

                for k, v in val_dict.items():
                    self.writer.add_scalar(f'val/{k}', v, epoch)

                if self.best_val < val_dict[self.best_val_metric]:
                    self.best_val = val_dict[self.best_val_metric]
                    self.best_val_epoch = epoch
                    self._save_checkpoint(epoch, 'best_val_checkpoint')


    def _save_checkpoint(self, epoch, filename='checkpoint'):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'best_val': self.best_val,
            'best_val_epoch': self.best_val_epoch,
            'best_val_metric': self.best_val_metric
        }
        filename = os.path.join(self.checkpoint_dir, f'{filename}.pth')
        # self.log.info("Saving checkpoint: {} ...".format(filename))
        torch.save(state, filename)


class ContrastiveLossTrainer(AlignmentTrainer):

    def __init__(self, config, train_dataloader, val_dataloader=None):
        if val_dataloader is not None:
            assert val_dataloader.batch_size == 1, "Val set batch size must be 1 for now."
        super().__init__(config, train_dataloader, val_dataloader)
        self.neg_thresh = config.neg_thresh
        self.pos_thresh = config.pos_thresh
        self.neg_weight = config.neg_weight

    def apply_transform(self, pts, trans):
        R = trans[:3, :3]
        T = trans[:3, 3]
        return pts @ R.t() + T

    def generate_rand_negative_pairs(self, positive_pairs, hash_seed, N0, N1, N_neg=0):
        """
    Generate random negative pairs
    """
        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)
        if N_neg < 1:
            N_neg = positive_pairs.shape[0] * 2
        pos_keys = misc._hash(positive_pairs, hash_seed)

        neg_pairs = np.floor(np.random.rand(int(N_neg), 2) * np.array([[N0, N1]])).astype(np.int64)
        neg_keys = misc._hash(neg_pairs, hash_seed)
        mask = np.isin(neg_keys, pos_keys, assume_unique=False)
        return neg_pairs[np.logical_not(mask)]

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0

        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()

        iter_size = self.iter_size
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)

        data_meter, data_timer, total_timer = timer.AverageMeter(), timer.Timer(), timer.Timer()

        # Main training
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            # for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                # Caffe iter size
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                # pairs consist of (xyz1 index, xyz0 index)
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1 = self.model(sinput1).F

                N0, N1 = len(sinput0), len(sinput1)

                pos_pairs = input_dict['correspondences']
                neg_pairs = self.generate_rand_negative_pairs(pos_pairs, max(N0, N1), N0, N1)
                pos_pairs = pos_pairs.long().to(self.device)
                neg_pairs = torch.from_numpy(neg_pairs).long().to(self.device)

                neg0 = F0.index_select(0, neg_pairs[:, 0])
                neg1 = F1.index_select(0, neg_pairs[:, 1])
                pos0 = F0.index_select(0, pos_pairs[:, 0])
                pos1 = F1.index_select(0, pos_pairs[:, 1])

                # Positive loss
                pos_loss = (pos0 - pos1).pow(2).sum(1)

                # Negative loss
                neg_loss = F.relu(self.neg_thresh - ((neg0 - neg1).pow(2).sum(1) + 1e-4).sqrt()).pow(2)

                pos_loss_mean = pos_loss.mean() / iter_size
                neg_loss_mean = neg_loss.mean() / iter_size

                # Weighted loss
                loss = pos_loss_mean + self.neg_weight * neg_loss_mean
                loss.backward()  # To accumulate gradient, zero gradients only at the begining of iter_size
                batch_loss += loss.item()
                batch_pos_loss += pos_loss_mean.item()
                batch_neg_loss += neg_loss_mean.item()

            self.optimizer.step()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            # Print logs
            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
                # self.log.info("Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}".format(epoch, curr_iter,
                #                                                                                                 len(self.data_loader) // iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
                #                  "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()

    def _valid_epoch(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        # torch.manual_seed(self.golden_seed)
        # np.random.seed(self.golden_seed)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        bar = tqdm(range(tot_num_data), colour='blue', unit='batch', leave=False)
        # for batch_idx in range(tot_num_data):
        for batch_idx in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device), requires_grad=False)
            soutput0 = self.model(sinput0)
            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device), requires_grad=False)
            soutput1 = self.model(sinput1)
            if self.model._get_name() == 'ResUNetBN2C':
                F0 = soutput0.F
                F1 = soutput1.F
            elif self.model._get_name() == 'DDPNetBN2C':
                F0 = soutput0[0].F
                F1 = soutput1[0].F
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)  # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

        self.log.info(
            f'epoch {epoch:3d}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HitRatio:{hit_ratio_meter.avg:.3f},FeatMatchRatio:{feat_match_ratio.avg:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg}

    def find_corr(self, xyz0, xyz1, F0, F1, subsample_size=-1):  # ([15864, 3]), ([19444, 3]), ([15864, 32]), ([19444, 32])
        subsample = len(F0) > subsample_size
        if subsample_size > 0 and subsample:
            N0 = min(len(F0), subsample_size)
            N1 = min(len(F1), subsample_size)
            inds0 = np.random.choice(len(F0), N0, replace=False)
            inds1 = np.random.choice(len(F1), N1, replace=False)
            F0, F1 = F0[inds0], F1[inds1]

        # Compute the nn
        nn_inds = pointcloud.find_nn_gpu(F0, F1, nn_max_n=self.config.nn_max_n)
        # nn_inds = knn.find_knn_gpu_cuml(F0, F1).reshape(-1).cpu()
        if subsample_size > 0 and subsample:
            return xyz0[inds0], xyz1[inds1[nn_inds]]
        else:
            return xyz0, xyz1[nn_inds]

    def evaluate_hit_ratio(self, xyz0, xyz1, T_gth, thresh=0.1):
        xyz0 = self.apply_transform(xyz0, T_gth)
        dist = np.sqrt(((xyz0 - xyz1)**2).sum(1) + 1e-6)  # ([5000])
        return (dist < thresh).float().mean().item()


class HCLTrainer(ContrastiveLossTrainer):

    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs, num_pos, num_hn_samples):
        """
        Generate negative pairs
        """
        assert positive_pairs[:,0].max() < F0.shape[0], 'match indices 0 overflow'
        assert positive_pairs[:,1].max() < F1.shape[0], 'match indices 1 overflow'

        hash_seed = max(len(F0), len(F1))

        # positive pair downsample
        if len(positive_pairs) > num_pos:
            pos_sel = np.random.choice(len(positive_pairs), num_pos, replace=False)  # 4096
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()  # Note pos_ind0 may have duplicate elements
        pos_ind1 = sample_pos_pairs[:, 1].long()

        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        # all downsample
        pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy())))  # B*1024
        # positive_pool = np.array(list(set(pos_ind0.cpu().numpy())))  # <= 4096
        # remain_pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy()) - set(pos_ind0.cpu().numpy())))  # >=1024 (e.g., 2362 = 5120 - 2758)
        # sel0 = np.random.choice(remain_pool, min(len(F0), num_hn_samples), replace=False)  # 1024

        # sel0 = np.random.choice(pool, min(len(F1), num_hn_samples), replace=False)        # V0, bug, but works for
        # sel1 = np.random.choice(len(F1), min(len(F1), num_hn_samples), replace=False)

        sel0 = np.random.choice(pool, min(len(pool), num_hn_samples), replace=False)  # 1024
        sel1 = np.random.choice(len(F1), min(len(pool), num_hn_samples), replace=False)

        subF0, subF1 = F0[sel0], F1[sel1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)  # ([4096]), ([4096])local index
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]  # global index
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)

        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size  # 1
        data_timer, total_timer, hc_timer, back_timer, opt_timer, forward_timer, clean_timer = \
        timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            # total_timer.tic()
            for iter_idx in range(iter_size):  # 1
                # data_timer.tic()
                input_dict = data_loader_iter.next()
                # data_timer.toc()

                # forward_timer.tic()
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device), requires_grad=False)
                soutput0 = self.model(sinput0)
                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device), requires_grad=False)
                soutput1 = self.model(sinput1)

                assert len(sinput0) == len(input_dict['sinput0_F']), 'diminish 0'
                assert len(sinput1) == len(input_dict['sinput1_F']), 'diminish 1'

                if self.model._get_name() == 'ResUNetBN2C':
                    F0 = soutput0.F
                    F1 = soutput1.F
                elif self.model._get_name() == 'DDPNetBN2C':
                    F0 = soutput0[0].F
                    F1 = soutput1[0].F
                # forward_timer.toc()

                # hc_timer.tic()
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0,
                    F1,
                    input_dict['correspondences'],
                    num_pos=self.config.num_pos_per_batch * self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size,
                )
                # hc_timer.toc()

                back_timer.tic()
                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()
                back_timer.toc()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            # opt_timer.tic()
            self.optimizer.step()
            # opt_timer.toc()

            # clean_timer.tic()
            gc.collect()
            torch.cuda.empty_cache()
            # clean_timer.toc()

            total_loss += batch_loss
            total_num += 1.0
            # total_timer.toc()

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
            # if curr_iter % 10 == 0:
            #     self.log.info(
            #         f'=========\ntotal {total_timer.avg:.2f} = data {data_timer.avg:.2f} + forward {forward_timer.avg:.2f} + hc {hc_timer.avg:.2f} + back {back_timer.avg:.2f} + opt {opt_timer.avg:.2f} + clean {clean_timer.avg:.2f}=========\n')
            #     forward_timer.reset()
            #     clean_timer.reset()
            #     back_timer.reset()
            #     opt_timer.reset()
            #     hc_timer.reset()
            #     data_timer.reset()
            #     total_timer.reset()


class HCLRTrainer(ContrastiveLossTrainer):

    def contrastive_hardest_negative_loss(self, F0, F1, V0, V1, positive_pairs, len_batch, num_pos=5192, num_hn_samples=2048, thresh=None):
        """
    Generate negative pairs
    """
        hash_seed = max(len(F0), len(F1))

        # positive pair downsample
        if len(positive_pairs) > num_pos:
            pos_sel = np.random.choice(len(positive_pairs), int(num_pos*1.1), replace=False)                # B*1024
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()           # note pos_ind0 may have duplicate source indices
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]          # B*1024
        posV0, posV1 = V0[pos_ind0], V1[pos_ind1]          # B*1024

        pairV = posV0 + posV1
        len_batch0 = torch.cumsum(torch.tensor([x[0] for x in len_batch]), dim=0)
        inds_global_keep = []
        for i in range(len(len_batch0)):
            if i == 0:
                mask_global = pos_ind0 < len_batch0[i] # 9011
            else:
                mask_global = torch.logical_and(pos_ind0 > len_batch0[i - 1], pos_ind0 < len_batch0[i]) # 9011
            inds_global = torch.arange(pos_ind0.shape[0])[mask_global]
            pairV_sub = pairV[mask_global]    # 710
            sorted_sub, inds_sub = torch.sort(pairV_sub, dim=0)   # in ascending order, ([N, 1]), ([N, 1])
            inds_global_keep.append(inds_global[inds_sub[:int(0.95 * inds_sub.shape[0])]])
        inds_global_keep = torch.cat(inds_global_keep)
        inds_sel = np.random.choice(len(inds_global_keep), int(num_pos), replace=False)                # B*1024
        inds_global_keep = inds_global_keep[inds_sel]
        maskV = torch.zeros((pos_ind0.shape[0], 1))
        maskV[inds_global_keep] = 1
        maskV = maskV > 0

        # all downsample
        pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy())))                                      # remove duplicate source indices
        positive_pool = np.array(list(set(pos_ind0.cpu().numpy())))                                         # <= 4096
        remain_pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy()) - set(pos_ind0.cpu().numpy()))) # >=1024 (e.g., 2362 = 5120 - 2758)
        # sel0 = np.random.choice(remain_pool, min(len(F0), num_hn_samples), replace=False)  # 1024
        sel0 = np.random.choice(pool, min(len(F0), num_hn_samples), replace=False)                 # B*256
        sel1 = np.random.choice(len(F1), min(len(F1), num_hn_samples), replace=False)              # B*256

        subF0, subF1 = F0[sel0], F1[sel1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)    # ([4096]), ([4096])local index
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]                # global index
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))

        pairV = posV0 + posV1
        sortedV, indsV = torch.sort(pairV, dim=0)   # in ascending order, ([B*1024, 1]), ([B*1024, 1])
        maskV = torch.ones_like(pairV).flatten()
        maskV[indsV[int(num_pos):]] = 0
        maskV = maskV == 1

        pos_loss = F.relu((posF0 - posF1)[maskV].pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)

        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size     # 1
        # data_timer, total_timer, hc_timer, back_timer, opt_timer, forward_timer, clean_timer = \
        # timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            # total_timer.tic()
            for iter_idx in range(iter_size):              # 1
                # data_timer.tic()
                input_dict = data_loader_iter.next()
                # data_timer.toc()

                # forward_timer.tic()
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device), requires_grad=False)
                soutput0 = self.model(sinput0)
                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device), requires_grad=False)
                soutput1 = self.model(sinput1)

                if self.model._get_name() == 'ResUNetBN2C':
                    F0 = soutput0.F
                    F1 = soutput1.F
                elif self.model._get_name() == 'DDPNetBN2C':
                    F0 = soutput0[0].F
                    F1 = soutput1[0].F
                # forward_timer.toc()

                # Auxiliary uncertainty estimation
                with torch.no_grad():
                    soutput0_aux = self.model_aux(sinput0)
                    soutput1_aux = self.model_aux(sinput1)
                    V0 = soutput0_aux[1].F
                    V1 = soutput1_aux[1].F

                # hc_timer.tic()
                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0, F1, V0, V1, pos_pairs, input_dict['len_batch'], \
                    num_pos=self.config.num_pos_per_batch * self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size,
                )
                # hc_timer.toc()

                # back_timer.tic()
                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()
                # back_timer.toc()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            # opt_timer.tic()
            self.optimizer.step()
            # opt_timer.toc()

            # clean_timer.tic()
            gc.collect()
            torch.cuda.empty_cache()
            # clean_timer.toc()

            total_loss += batch_loss
            total_num += 1.0
            # total_timer.toc()

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
            # if curr_iter % 10 == 0:
            #     self.log.info(
            #         f'total {total_timer.avg:.2f} = data {data_timer.avg:.2f} + forward {forward_timer.avg:.2f} + hc {hc_timer.avg:.2f} + back {back_timer.avg:.2f} + opt {opt_timer.avg:.2f} + clean {clean_timer.avg:.2f}')
            #     forward_timer.reset()
            #     clean_timer.reset()
            #     back_timer.reset()
            #     opt_timer.reset()
            #     hc_timer.reset()
            #     data_timer.reset()
            #     total_timer.reset()


class HCRTrainer(ContrastiveLossTrainer):
    def __init__(self, config, data_loader, val_data_loader=None):
        super().__init__(config, data_loader, val_data_loader)


    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs, num_pos=5192, num_hn_samples=2048, thresh=None):
        hash_seed = max(len(F0), len(F1))

        # positive pair downsample
        if len(positive_pairs) > num_pos:
            pos_sel = np.random.choice(len(positive_pairs), num_pos, replace=False)  # 4096
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()  # Note pos_ind0 may have duplicate elements
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        # all downsample
        pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy())))  # 5120
        positive_pool = np.array(list(set(pos_ind0.cpu().numpy())))  # <= 4096
        remain_pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy()) - set(pos_ind0.cpu().numpy())))  # >=1024 (e.g., 2362 = 5120 - 2758)
        # sel0 = np.random.choice(remain_pool, min(len(F0), num_hn_samples), replace=False)  # 1024
        sel0 = np.random.choice(pool, min(len(F0), num_hn_samples), replace=False)  # 1024
        sel1 = np.random.choice(len(F1), min(len(F1), num_hn_samples), replace=False)

        subF0, subF1 = F0[sel0], F1[sel1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)  # ([4096]), ([4096])local index
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]  # global index
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)

        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size  # 1
        data_meter, data_timer, total_timer = timer.AverageMeter(), timer.Timer(), timer.Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)

        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            # for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):  # 1
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                # [1/2] metric learning loss
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device), requires_grad=False)
                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device), requires_grad=False)
                F0 = self.model(sinput0).F
                F1 = self.model(sinput1).F
                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0, F1, pos_pairs,
                    num_pos=self.config.num_pos_per_batch * self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size,
                )
                pos_loss /= iter_size
                neg_loss /= iter_size
                loss_metric = pos_loss + self.neg_weight * neg_loss

                # [2/2] Guassian Newton Optimisation Loss
                loss_registration = self.solver(input_dict)

                loss = loss_metric + loss_registration
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/lossm', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/lossm_positive', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/lossm_negative', batch_neg_loss, start_iter + curr_iter)
                data_meter.reset()
                total_timer.reset()

            # inspect stochastic method outputs
            img_tensor = self.solver.F_res_track # [10, 32]
            img_tensor = (img_tensor - img_tensor.min())/(img_tensor.max() - img_tensor.min())*255
            img_np = img_tensor.type(torch.ByteTensor).cpu().detach().numpy()
            fig, axs = plt.subplots(1, 1, figsize=(5, 5), squeeze=False)
            ax = axs[0][0]
            ax.imshow(img_np, cmap='Greys')
            matin.ax_default_style(ax)
            matin.ax_lims(ax, interval_xticks=8, interval_yticks=2)
            plt.savefig(f'F_ref_{start_iter + curr_iter}.png', dpi=200)
            pass




class BayesianHardestContrastiveLossTrainer(ContrastiveLossTrainer):

    def bayesian_hardest_contrastive_loss(self, F0, V0, F1, V1, positive_pairs, num_pos=5192, num_hn_samples=2048, thresh=None):
        """Generate negative pairs"""
        hash_seed = max(len(F0), len(F1))

        # positive pair downsample
        if len(positive_pairs) > num_pos:
            pos_sel = np.random.choice(len(positive_pairs), num_pos, replace=False)  # 4096
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()  # Note pos_ind0 may have duplicate elements
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]
        posV0, posV1 = V0[pos_ind0], V1[pos_ind1]

        # all downsample
        pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy())))  # 5120
        positive_pool = np.array(list(set(pos_ind0.cpu().numpy())))  # <= 4096
        remain_pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy()) - set(pos_ind0.cpu().numpy())))  # >=1024 (e.g., 2362 = 5120 - 2758)
        # sel0 = np.random.choice(remain_pool, min(len(F0), num_hn_samples), replace=False)  # 1024
        sel0 = np.random.choice(pool, min(len(F0), num_hn_samples), replace=False)  # 1024
        sel1 = np.random.choice(len(F1), min(len(F1), num_hn_samples), replace=False)

        subF0, subF1 = F0[sel0], F1[sel1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        # pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        # neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        # neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)

        muA0, muP0, muN0 = posF0, posF1, F1[D01ind[mask0]]
        varA0, varP0, varN0 = posV0, posV1, V1[D01ind[mask0]]
        muA1, muP1, muN1 = posF1, posF0, F0[D10ind[mask1]]
        varA1, varP1, varN1 = posV1, posV0, V0[D10ind[mask1]]

        pos_loss, pos_prob, pos_mu, pos_sigma = self.bayesian_contrastive_loss(muA0, muP0, varA0, varP0, is_pos=True)
        neg_loss0, neg_prob0, neg_mu0, neg_sigma0 = self.bayesian_contrastive_loss(muA0[mask0], muN0, varA0[mask0], varN0, is_pos=False)
        neg_loss1, neg_prob1, neg_mu1, neg_sigma1 = self.bayesian_contrastive_loss(muA1[mask1], muN1, varA1[mask1], varN1, is_pos=False)
        if self.curr_iter % self.config.stat_every_iter == 0:
            self.writer.add_scalar('positive/pos_loss', pos_loss.item(), self.curr_iter)
            self.writer.add_scalar('positive/pos_prob', pos_prob.item(), self.curr_iter)
            self.writer.add_scalar('positive/pos_mu', pos_mu.item(), self.curr_iter)
            self.writer.add_scalar('positive/pos_sigma', pos_sigma.item(), self.curr_iter)
            self.writer.add_scalar('positive/mp', self.pos_thresh, self.curr_iter)
            self.writer.add_scalar('negative0/neg_loss0', neg_loss0.item(), self.curr_iter)
            self.writer.add_scalar('negative0/neg_prob0', neg_prob0.item(), self.curr_iter)
            self.writer.add_scalar('negative0/neg_mu0', neg_mu0.item(), self.curr_iter)
            self.writer.add_scalar('negative0/neg_sigma0', neg_sigma0.item(), self.curr_iter)
            self.writer.add_scalar('negative0/mn', self.neg_thresh, self.curr_iter)
            self.writer.add_scalar('negative1/neg_loss1', neg_loss1.item(), self.curr_iter)
            self.writer.add_scalar('negative1/neg_prob1', neg_prob1.item(), self.curr_iter)
            self.writer.add_scalar('negative1/neg_mu1', neg_mu1.item(), self.curr_iter)
            self.writer.add_scalar('negative1/neg_sigma1', neg_sigma1.item(), self.curr_iter)

        return pos_loss, (neg_loss0 + neg_loss1) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = timer.AverageMeter(), timer.Timer(), timer.Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)

        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            self.curr_iter = curr_iter + start_iter
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0, V0 = self.model(sinput0)

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1, V1 = self.model(sinput1)

                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.bayesian_hardest_contrastive_loss(
                    F0,
                    V0,
                    F1,
                    V1,
                    pos_pairs,
                    num_pos=self.config.num_pos_per_batch * self.config.batch_size,
                    num_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size,
                )

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
                data_meter.reset()
                total_timer.reset()

    def _valid_epoch(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        bar = tqdm(range(tot_num_data), colour='blue', unit='batch', leave=False)
        # for batch_idx in range(tot_num_data):
        for batch_idx in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
            F0, V0 = self.model(sinput0)

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
            F1, V1 = self.model(sinput1)
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)  # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

            # if batch_idx % 100 == 0 and batch_idx > 0:
            #   self.log.info(' '.join([
            #       f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},", f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},",
            #       f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},", f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
            #   ]))
            #   data_timer.reset()

        self.log.info(
            f'epoch {epoch}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HitRatio:{hit_ratio_meter.avg:.3f},FeatMatchRatio:{feat_match_ratio.avg:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg}


class TripletLossTrainer(ContrastiveLossTrainer):

    def triplet_loss(self, F0, F1, positive_pairs, num_pos=1024, num_hn_samples=None, num_rand_triplet=1024):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        # unused actually
        if num_pos_pairs > num_pos:  # num_pos=256
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)  # num_rand_triplet=1024
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

        loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()

        return loss, pos_dist.mean(), rand_neg_dist.mean()

    def _train_epoch(self, epoch):
        config = self.config

        gc.collect()
        self.model.train()

        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = timer.AverageMeter(), timer.Timer(), timer.Timer()
        pos_dist_meter, neg_dist_meter = timer.AverageMeter(), timer.AverageMeter()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)

        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            batch_loss = 0
            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                data_time += data_timer.toc(average=False)
                input_dict = data_loader_iter.next()
                # pairs consist of (xyz1 index, xyz0 index)
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1 = self.model(sinput1).F

                pos_pairs = input_dict['correspondences']
                loss, pos_dist, neg_dist = self.triplet_loss(
                    F0,
                    F1,
                    pos_pairs,
                    num_pos=config.triplet_num_pos * config.batch_size,  # 256
                    num_hn_samples=config.triplet_num_hn * config.batch_size,  # 512
                    num_rand_triplet=config.triplet_num_rand * config.batch_size)  # 1024
                loss /= iter_size
                loss.backward()
                batch_loss += loss.item()
                pos_dist_meter.update(pos_dist)
                neg_dist_meter.update(neg_dist)

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                # self.log.info("Train Epoch: {} [{}/{}], Current Loss: {:.3e}, Pos dist: {:.3e}, Neg dist: {:.3e}".format(epoch, curr_iter,
                #                                                                                                         len(self.data_loader) //
                #                                                                                                         iter_size, batch_loss, pos_dist_meter.avg, neg_dist_meter.avg) +
                #              "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
                pos_dist_meter.reset()
                neg_dist_meter.reset()
                data_meter.reset()
                total_timer.reset()


class HardestTripletLossTrainer(TripletLossTrainer):
    # num_pos=4*256, num_hn_samples=4*512, num_rand_triplet=4*1024
    def triplet_loss(self, F0, F1, positive_pairs, num_pos=1024, num_hn_samples=512, num_rand_triplet=1024):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        # positive pairs downsample
        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]  # ([4*256])
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]  # ([4*256])

        # all downsample
        pool = np.array(list(set(positive_pairs[:, 0].cpu().numpy())))
        # sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)  # modify this line to accelerate training
        sel0 = np.random.choice(pool, min(N0, num_hn_samples), replace=False)  # 4*512
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        subF0, subF1 = F0[sel0], F1[sel1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')  # D01: (4*256, 4*512)
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)  # ([4*256]), ([4*256])
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)  # ([4*256])
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)  # ([4*256])

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))  # ([4*256])
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)  # ([4*256])

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)  # num_rand_triplet=1024
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

        loss = F.relu(torch.cat([rand_pos_dist + self.neg_thresh - rand_neg_dist, pos_dist[mask0] + self.neg_thresh - D01min[mask0], pos_dist[mask1] + self.neg_thresh - D10min[mask1]])).mean()

        return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2


class BTLTrainer(ContrastiveLossTrainer):

    def triplet_loss(self, F0, V0, F1, V1, positive_pairs, num_pos=1024, num_rand_triplet=1024):
        '''
        Generate negative pairs
        '''
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        if N1 < num_rand_triplet :
            # print(f'num_pos_pairs:{num_pos_pairs}, num_rand_triplet:{num_rand_triplet}, N1:{N1}')
            num_rand_triplet = N1
        if num_pos_pairs < num_rand_triplet:
            num_rand_triplet = num_pos_pairs
            # print(f'num_pos_pairs:{num_pos_pairs}, num_rand_triplet:{num_rand_triplet}, N1:{N1}')

        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)
        # loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()
        # return loss, pos_dist.mean(), rand_neg_dist.mean()

        loss, probs, mu, sigma = self.bayesian_triplet_loss(F0[anchors], F1[positives], F1[negatives], V0[anchors], V1[positives], V1[negatives])
        if self.curr_iter % 5 == 0:
            self.writer.add_scalar('triplet/pos_dist', rand_pos_dist.mean().item(), self.curr_iter)
            self.writer.add_scalar('triplet/neg_dist', rand_neg_dist.mean().item(), self.curr_iter)
            self.writer.add_scalar('triplet/loss', loss.item(), self.curr_iter)
            self.writer.add_scalar('triplet/probs', probs.item(), self.curr_iter)
            self.writer.add_scalar('triplet/mu', mu.item(), self.curr_iter)
            self.writer.add_scalar('triplet/sigma', sigma.item(), self.curr_iter)
            self.writer.add_scalar('triplet/margin', self.config.bayesian_margin, self.curr_iter)
        return loss, 0, 0

    def _train_epoch(self, epoch):
        config = self.config

        gc.collect()
        if self.config.phase == 'train_sigma':
            self.model.eval()
        else:
            self.model.train()

        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        pos_dist_meter, neg_dist_meter = timer.AverageMeter(), timer.AverageMeter()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        # bar = tqdm(range(50), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            self.curr_iter = curr_iter + start_iter
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            batch_loss = 0
            data_time = 0
            for iter_idx in range(iter_size):

                input_dict = data_loader_iter.next()

                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0, V0 = self.model(sinput0)

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1, V1 = self.model(sinput1)

                loss, pos_dist, neg_dist = self.triplet_loss(
                    F0.F, V0.F, F1.F, V1.F, input_dict['correspondences'], \
                    num_pos=config.triplet_num_pos * config.batch_size,
                    num_rand_triplet=config.triplet_num_rand * config.batch_size)
                loss /= iter_size
                loss.backward()
                batch_loss += loss.item()
                pos_dist_meter.update(pos_dist)
                neg_dist_meter.update(neg_dist)

            self.optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                pos_dist_meter.reset()
                neg_dist_meter.reset()

    def _valid_epoch_bak(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        # torch.manual_seed(self.golden_seed)   # training will be stuck if we introduce determinstic behaviour here
        # np.random.seed(self.golden_seed)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data) # 400, 643
        data_loader_iter = self.val_data_loader.__iter__()
        for batch_idx in tqdm(range(tot_num_data), colour='blue', unit='batch', leave=False):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
            F0, V0 = self.model(sinput0)

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
            F1, V1 = self.model(sinput1)
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0.F, F1.F, subsample_size=5000)  # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

        self.log.info(
            f'epoch {epoch}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HitRatio:{hit_ratio_meter.avg:.3f},FeatMatchRatio:{feat_match_ratio.avg:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg}

    def _valid_epoch(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        # torch.manual_seed(self.golden_seed)   # training will be stuck if we introduce determinstic behaviour here
        # np.random.seed(self.golden_seed)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data) # 400, 643
        data_loader_iter = self.val_data_loader.__iter__()

        bin_hit_ratios_pairs = np.zeros((tot_num_data, 10))
        bin_hit_ratios_pairs_counts = np.zeros((tot_num_data, 10))
        hit_ratios = np.zeros((tot_num_data, 1))

        inds_to_vis = np.linspace(0, tot_num_data, 15).astype(np.int32)
        for batch_idx in tqdm(range(tot_num_data), colour='blue', unit='batch', leave=False):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
            F0, V0 = self.model(sinput0)

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
            F1, V1 = self.model(sinput1)
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0.F, F1.F, subsample_size=5000)  # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            bin_hit_ratios_pair, bin_counts_pair, hr = \
                uncertainty_util.parse_hr_uncertainty(feat=F0.F, feat_target=F1.F, xyz=xyz0, xyz_target=xyz1, sigma=V0.F, sigma_target=V1.F, T_gt=T_gt, tau1=self.config.hit_ratio_thresh)
            bin_hit_ratios_pairs[batch_idx] = bin_hit_ratios_pair
            bin_hit_ratios_pairs_counts[batch_idx] = bin_counts_pair
            hit_ratios[batch_idx] = hr

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()
            if batch_idx in inds_to_vis:
                self.writer.add_histogram(f'uncertainty_distribution/{batch_idx}', V0.F, epoch)


        self.writer.add_figure('ece/hr-uncertainty', uncertainty_util.output_hr_bins(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios.mean()), epoch)
        self.writer.add_figure('ece/fmr-threshold-bin', uncertainty_util.output_fmr_threshold_bins(bin_hit_ratios_pairs, hit_ratios), epoch)
        self.writer.add_figure('ece/fmr-threshold-bins', uncertainty_util.output_fmr_threshold_bins_avg(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios, [1, 2, 3, 4, 5, 6]), epoch)

        self.writer.flush()

        bin_hit_ratios_pair_avg = bin_hit_ratios_pairs.mean(axis=0, keepdims=True)[0]
        bin_hit_ratios_pairs_counts_avg = bin_hit_ratios_pairs_counts.mean(axis=0, keepdims=True)[0]
        ece = uncertainty_util.cal_ece(bin_hit_ratios_pair_avg, bin_hit_ratios_pairs_counts_avg)

        self.log.info(
            f'epoch {epoch:<3d}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HR:{hit_ratio_meter.avg:.3f},FMR:{feat_match_ratio.avg:.3f},ECE:{ece:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg, 'ece': ece}


class BTLSTrainer(ContrastiveLossTrainer):

    def triplet_loss(self, F0, V0, F1, V1, positive_pairs, num_pos=1024, num_rand_triplet=1024):
        '''
        Generate negative pairs
        '''
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        if N1 < num_rand_triplet :
            # print(f'num_pos_pairs:{num_pos_pairs}, num_rand_triplet:{num_rand_triplet}, N1:{N1}')
            num_rand_triplet = N1
        if num_pos_pairs < num_rand_triplet:
            num_rand_triplet = num_pos_pairs
            # print(f'num_pos_pairs:{num_pos_pairs}, num_rand_triplet:{num_rand_triplet}, N1:{N1}')

        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)
        # loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()
        # return loss, pos_dist.mean(), rand_neg_dist.mean()

        loss = self.bayesian_triplet_loss_sampling(F0[anchors], F1[positives], F1[negatives], V0[anchors], V1[positives], V1[negatives])
        return loss, 0, 0

    def _train_epoch(self, epoch):
        config = self.config

        gc.collect()
        if self.config.phase == 'train_sigma':
            self.model.eval()
        else:
            self.model.train()

        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        pos_dist_meter, neg_dist_meter = timer.AverageMeter(), timer.AverageMeter()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        # bar = tqdm(range(50), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            self.curr_iter = curr_iter + start_iter
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            batch_loss = 0
            data_time = 0
            for iter_idx in range(iter_size):

                input_dict = data_loader_iter.next()

                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0, V0 = self.model(sinput0)

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1, V1 = self.model(sinput1)

                loss, pos_dist, neg_dist = self.triplet_loss(
                    F0.F, V0.F, F1.F, V1.F, input_dict['correspondences'], \
                    num_pos=config.triplet_num_pos * config.batch_size,
                    num_rand_triplet=config.triplet_num_rand * config.batch_size)
                loss /= iter_size
                loss.backward()
                batch_loss += loss.item()
                pos_dist_meter.update(pos_dist)
                neg_dist_meter.update(neg_dist)

            self.optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                pos_dist_meter.reset()
                neg_dist_meter.reset()


    def _valid_epoch(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        # torch.manual_seed(self.golden_seed)   # training will be stuck if we introduce determinstic behaviour here
        # np.random.seed(self.golden_seed)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data) # 400, 643
        data_loader_iter = self.val_data_loader.__iter__()

        bin_hit_ratios_pairs = np.zeros((tot_num_data, 10))
        bin_hit_ratios_pairs_counts = np.zeros((tot_num_data, 10))
        hit_ratios = np.zeros((tot_num_data, 1))

        inds_to_vis = np.linspace(0, tot_num_data, 15).astype(np.int32)
        for batch_idx in tqdm(range(tot_num_data), colour='blue', unit='batch', leave=False):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
            F0, V0 = self.model(sinput0)

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
            F1, V1 = self.model(sinput1)
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0.F, F1.F, subsample_size=5000)  # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            bin_hit_ratios_pair, bin_counts_pair, hr = \
                uncertainty_util.parse_hr_uncertainty(feat=F0.F, feat_target=F1.F, xyz=xyz0, xyz_target=xyz1, sigma=V0.F, sigma_target=V1.F, T_gt=T_gt, tau1=self.config.hit_ratio_thresh)
            bin_hit_ratios_pairs[batch_idx] = bin_hit_ratios_pair
            bin_hit_ratios_pairs_counts[batch_idx] = bin_counts_pair
            hit_ratios[batch_idx] = hr

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()
            if batch_idx in inds_to_vis:
                self.writer.add_histogram(f'uncertainty_distribution/{batch_idx}', V0.F, epoch)


        self.writer.add_figure('ece/hr-uncertainty', uncertainty_util.output_hr_bins(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios.mean()), epoch)
        self.writer.add_figure('ece/fmr-threshold-bin', uncertainty_util.output_fmr_threshold_bins(bin_hit_ratios_pairs, hit_ratios), epoch)
        self.writer.add_figure('ece/fmr-threshold-bins', uncertainty_util.output_fmr_threshold_bins_avg(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios, [1, 2, 3, 4, 5, 6]), epoch)

        self.writer.flush()

        bin_hit_ratios_pair_avg = bin_hit_ratios_pairs.mean(axis=0, keepdims=True)[0]
        bin_hit_ratios_pairs_counts_avg = bin_hit_ratios_pairs_counts.mean(axis=0, keepdims=True)[0]
        ece = uncertainty_util.cal_ece(bin_hit_ratios_pair_avg, bin_hit_ratios_pairs_counts_avg)

        self.log.info(
            f'epoch {epoch:<3d}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HR:{hit_ratio_meter.avg:.3f},FMR:{feat_match_ratio.avg:.3f},ECE:{ece:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg, 'ece': ece}


class BTRTrainer(ContrastiveLossTrainer):

    def triplet_loss(self, F0, V0, F1, V1, positive_pairs, num_pos=1024, num_hn_samples=None, num_rand_triplet=1024):
        '''
        Generate negative pairs
        '''
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)
        # loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()
        # return loss, pos_dist.mean(), rand_neg_dist.mean()

        loss, probs, mu, sigma = self.bayesian_triplet_loss(F0[anchors], F1[positives], F1[negatives], V0[anchors], V1[positives], V1[negatives])
        if self.curr_iter % 5 == 0:
            self.writer.add_scalar('triplet/pos_dist', rand_pos_dist.mean().item(), self.curr_iter)
            self.writer.add_scalar('triplet/neg_dist', rand_neg_dist.mean().item(), self.curr_iter)
            self.writer.add_scalar('triplet/loss', loss.item(), self.curr_iter)
            self.writer.add_scalar('triplet/probs', probs.item(), self.curr_iter)
            self.writer.add_scalar('triplet/mu', mu.item(), self.curr_iter)
            self.writer.add_scalar('triplet/sigma', sigma.item(), self.curr_iter)
            self.writer.add_scalar('triplet/margin', self.neg_thresh, self.curr_iter)

        return loss, 0, 0

    def _train_epoch(self, epoch):
        config = self.config

        gc.collect()
        self.model.train()

        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        total_timer, data_timer, metric_timer, reg1_timer, reg2_timer, reg3_timer= timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer(), timer.Timer()
        pos_dist_meter, neg_dist_meter = timer.AverageMeter(), timer.AverageMeter()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            self.curr_iter = curr_iter + start_iter
            bar.set_description(f'{self.checkpoint_dir}')
            self.optimizer.zero_grad()
            skip = 0
            batch_loss = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_timer.toc()

                metric_timer.tic()
                # ------------------- metric learning loss ------------------- #
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0, V0 = self.model(sinput0)
                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1, V1 = self.model(sinput1)

                loss_metric, pos_dist, neg_dist = self.triplet_loss(
                    F0.F, V0.F, F1.F, V1.F, input_dict['correspondences'], \
                    num_pos=config.triplet_num_pos * config.batch_size,
                    num_hn_samples=config.triplet_num_hn * config.batch_size,
                    num_rand_triplet=config.triplet_num_rand * config.batch_size)
                loss_metric /= iter_size
                metric_timer.toc()

                # --------------------- procrustes loss -------------------- #
                reg1_timer.tic()
                pred_pairs = self.find_pairs(F0.F, F1.F, input_dict['len_batch'])              # ([N0, 32])
                reg1_timer.toc()
                reg2_timer.tic()
                is_correct = misc.find_correct_correspondence(input_dict['correspondences_indepent'], pred_pairs, len_batch=input_dict['len_batch'])  # ([N0, 1])
                reg2_timer.toc()
                p0_lens = [x[0] for x in input_dict['len_batch']]
                p1_lens = [x[1] for x in input_dict['len_batch']]
                xyz0s = torch.split(input_dict['pcd0'], p0_lens)
                xyz1s = torch.split(input_dict['pcd1'], p1_lens)
                T_gts = torch.split(input_dict['T_gt'], len(p0_lens))
                T_gts = torch.stack(T_gts)
                reg3_timer.tic()
                pred_rots, pred_trans, w_sum = self.weighted_procrustes_batch(xyz0s=xyz0s, xyz1s=xyz1s, pred_pairs=pred_pairs, V0=V0, V1=V1, eps=config.eps)
                reg3_timer.toc()
                gt_rots, gt_trans = metrics.decompose_rotation_translation(T_gts)
                rot_error = metrics.batch_rotation_error(pred_rots, gt_rots)
                trans_error = metrics.batch_translation_error(pred_trans, gt_trans)
                individual_loss = rot_error + self.config.trans_weight * trans_error
                # valid_mask = w_sum < 1000  # TODO Necessary to apply a mask?
                # num_valid = valid_mask.sum().item()
                # loss_procrustes = individual_loss[valid_mask].mean()
                loss_procrustes = individual_loss.mean()
                loss_procrustes /= iter_size

                if not np.isfinite(loss_procrustes.item()):
                    self.nan_loss_count += 1
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                loss = loss_metric + config.procrustes_weight * loss_procrustes

                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 80, norm_type=2.0)


                para_grad_mean = 0
                for p in self.model.parameters():
                    para_grad_mean += p.grad.mean().item()
                    if torch.any(torch.isnan(p.grad)):
                        self.nan_grad_count += 1
                        skip=1
                        gc.collect()
                        torch.cuda.empty_cache()
                        print('nan grad +1')
                        continue
                print(para_grad_mean)

                # print(curr_iter, self.model.conv1.kernel.grad.mean().item(), \
                #         self.model.conv4.kernel.grad.mean().item(),\
                #         self.model.conv1_tr.kernel.grad.mean().item(),\
                #         self.model.sigma_conv1_tr.kernel.grad.mean().item())

                batch_loss += loss.item()
                pos_dist_meter.update(pos_dist)
                neg_dist_meter.update(neg_dist)

            if skip:
                gc.collect()
                torch.cuda.empty_cache()
                continue

            self.optimizer.step()
            gc.collect()
            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/nan_loss_count', self.nan_loss_count, start_iter + curr_iter)
                self.writer.add_scalar('train/nan_grad_count', self.nan_grad_count, start_iter + curr_iter)
                # total=0.62, data=0.19, metric=0.13, reg1=0.09, reg2=0.02, reg3=0.04
                # self.log.info(f'total={total_timer.avg:.2f}, data={data_timer.avg:.2f}, metric={metric_timer.avg:.2f}, reg1={reg1_timer.avg:.2f}, reg2={reg2_timer.avg:.2f}, reg3={reg3_timer.avg:.2f}')
                pos_dist_meter.reset()
                neg_dist_meter.reset()

    def _valid_epoch(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()
        for batch_idx in tqdm(range(tot_num_data), colour='blue', unit='batch', leave=False):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
            F0, V0 = self.model(sinput0)

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
            F1, V1 = self.model(sinput1)
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0.F, F1.F, subsample_size=5000)         # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

        self.log.info(
            f'epoch {epoch}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HitRatio:{hit_ratio_meter.avg:.3f},FeatMatchRatio:{feat_match_ratio.avg:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg}


    def weighted_procrustes_batch(self, xyz0s, xyz1s, V0, V1, pred_pairs, eps):
        def scale_v(var):
            var = (var - var.min()) / (var.max() - var.min() + 1e-9)
            var = 1 - var
            return var

        # decomposed_weights = self.decompose_by_length(weights, pred_pairs)
        _, decomposed_V0 = V0.decomposed_coordinates_and_features
        _, decomposed_V1 = V1.decomposed_coordinates_and_features

        RT = []
        weights_sum = []

        for xyz0, xyz1, pred_pair, v0, v1 in zip(xyz0s, xyz1s, pred_pairs, decomposed_V0, decomposed_V1):
            xyz0.requires_grad = False
            xyz1.requires_grad = False
            v0 = scale_v(v0[pred_pair[:, 0]])
            v1 = scale_v(v1[pred_pair[:, 1]])
            weights = v0 + v1
            weights_sum.append(weights.sum().item())
            predT = registration.weighted_procrustes(X=xyz0[pred_pair[:, 0]].to(self.device), Y=xyz1[pred_pair[:, 1]].to(self.device), w=weights, eps=eps)
            RT.append(predT)

        Rs, ts = list(zip(*RT))
        Rs = torch.stack(Rs, 0)
        ts = torch.stack(ts, 0)
        weights_sum = torch.Tensor(weights_sum)
        return Rs, ts, weights_sum

    def find_pairs(self, F0, F1, len_batch):
        nn_batch = knn.find_knn_batch(F0, F1, len_batch, nn_max_n=250, knn=1, return_distance=False, search_method='gpu')

        pred_pairs = []
        for nns, lens in zip(nn_batch, len_batch):
            pred_pair_ind0, pred_pair_ind1 = torch.arange(len(nns)).long()[:, None], nns.long().cpu()
            nn_pairs = []
            for j in range(nns.shape[1]):
                nn_pairs.append(torch.cat((pred_pair_ind0.cpu(), pred_pair_ind1[:, j].unsqueeze(1)), 1))

            pred_pairs.append(torch.cat(nn_pairs, 0))
        return pred_pairs



class BayesianContrastiveLossTrainer(ContrastiveLossTrainer):

    def contrastive_loss(self, F0, V0, F1, V1, positive_pairs, num_pos=1024, num_hn_samples=None, num_rand_triplet=1024):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        # rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        # rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)
        # loss = F.relu(rand_pos_dist + self.neg_thresh - rand_neg_dist).mean()
        # return loss, pos_dist.mean(), rand_neg_dist.mean()

        pos_loss, pos_prob, pos_mu, pos_sigma = self.bayesian_contrastive_loss(F0[anchors], F1[positives], V0[anchors], V1[positives], is_pos=True)
        # neg_loss, neg_prob, neg_mu, neg_sigma = self.bayesian_contrastive_loss(F0[anchors], F1[negatives], V0[anchors], V1[negatives], is_pos=False)
        # loss = (pos_loss + neg_loss) / 2
        loss = pos_loss
        if self.curr_iter % 5 == 0:
            self.writer.add_scalar('positive/pos_loss', pos_loss.item(), self.curr_iter)
            self.writer.add_scalar('positive/pos_prob', pos_prob.item(), self.curr_iter)
            self.writer.add_scalar('positive/pos_mu', pos_mu.item(), self.curr_iter)
            self.writer.add_scalar('positive/pos_sigma', pos_sigma.item(), self.curr_iter)
            self.writer.add_scalar('positive/mp', self.pos_thresh, self.curr_iter)
            # self.writer.add_scalar('negative/neg_loss', neg_loss.item(), self.curr_iter)
            # self.writer.add_scalar('negative/neg_prob', neg_prob.item(), self.curr_iter)
            # self.writer.add_scalar('negative/neg_mu', neg_mu.item(), self.curr_iter)
            # self.writer.add_scalar('negative/neg_sigma', neg_sigma.item(), self.curr_iter)
            # self.writer.add_scalar('negative/mn', self.neg_thresh, self.curr_iter)

        # loss = self.bayesian_triplet_loss(F0[anchors], F1[positives], F1[negatives], V0[anchors], V1[positives], V1[negatives])

        return loss, 0, 0

    def _train_epoch(self, epoch):
        config = self.config

        gc.collect()
        self.model.train()

        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = timer.AverageMeter(), timer.Timer(), timer.Timer()
        pos_dist_meter, neg_dist_meter = timer.AverageMeter(), timer.AverageMeter()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        # for curr_iter in range(len(data_loader) // iter_size):
        bar = tqdm(range(len(data_loader) // iter_size), colour='blue', unit='batch', leave=False)
        for curr_iter in bar:
            bar.set_description(f'{self.checkpoint_dir}')
            self.curr_iter = curr_iter + start_iter
            self.optimizer.zero_grad()
            batch_loss = 0
            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                data_time += data_timer.toc(average=False)
                input_dict = data_loader_iter.next()
                # pairs consist of (xyz1 index, xyz0 index)
                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0, V0 = self.model(sinput0)

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
                F1, V1 = self.model(sinput1)

                pos_pairs = input_dict['correspondences']
                loss, pos_dist, neg_dist = self.contrastive_loss(
                    F0, V0, F1, V1, pos_pairs, \
                    num_pos=config.triplet_num_pos * config.batch_size,
                    num_hn_samples=config.triplet_num_hn * config.batch_size,
                    num_rand_triplet=config.triplet_num_rand * config.batch_size)
                loss /= iter_size
                loss.backward()
                batch_loss += loss.item()
                pos_dist_meter.update(pos_dist)
                neg_dist_meter.update(neg_dist)

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                pos_dist_meter.reset()
                neg_dist_meter.reset()
                data_meter.reset()
                total_timer.reset()

    def _valid_epoch(self, epoch, lr):
        # Change the network to evaluation mode
        self.model.eval()
        self.val_data_loader.dataset.reset_seed(0)
        num_data = 0
        hit_ratio_meter, feat_match_ratio, loss_meter, rte_meter, rre_meter = timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter(), timer.AverageMeter()
        data_timer, feat_timer, matching_timer = timer.Timer(), timer.Timer(), timer.Timer()
        tot_num_data = len(self.val_data_loader.dataset)
        if self.val_max_iter > 0:
            tot_num_data = min(self.val_max_iter, tot_num_data)
        data_loader_iter = self.val_data_loader.__iter__()

        for batch_idx in range(tot_num_data):
            data_timer.tic()
            input_dict = data_loader_iter.next()
            data_timer.toc()

            # pairs consist of (xyz1 index, xyz0 index)
            feat_timer.tic()
            sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
            F0, V0 = self.model(sinput0)

            sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))
            F1, V1 = self.model(sinput1)
            feat_timer.toc()

            matching_timer.tic()
            xyz0, xyz1, T_gt = input_dict['pcd0'], input_dict['pcd1'], input_dict['T_gt']
            xyz0_corr, xyz1_corr = self.find_corr(xyz0, xyz1, F0, F1, subsample_size=5000)  # randomly sampled ([5000, 3]), ([5000, 3])
            T_est = transform_estimation.est_quad_linear_robust(xyz0_corr, xyz1_corr)

            loss = metrics.corr_dist(T_est, T_gt, xyz0, xyz1, weight=None)
            loss_meter.update(loss)

            rte = np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            rte_meter.update(rte)
            rre = np.arccos((np.trace(T_est[:3, :3].t() @ T_gt[:3, :3]) - 1) / 2)
            if not np.isnan(rre):
                rre_meter.update(rre)

            hit_ratio = self.evaluate_hit_ratio(xyz0_corr, xyz1_corr, T_gt, thresh=self.config.hit_ratio_thresh)
            hit_ratio_meter.update(hit_ratio)
            feat_match_ratio.update(hit_ratio > 0.05)
            matching_timer.toc()

            num_data += 1
            torch.cuda.empty_cache()

            # if batch_idx % 100 == 0 and batch_idx > 0:
            #     self.log.info(' '.join([
            #         f"Validation iter {num_data} / {tot_num_data} : Data Loading Time: {data_timer.avg:.3f},",
            #         f"Feature Extraction Time: {feat_timer.avg:.3f}, Matching Time: {matching_timer.avg:.3f},", f"Loss: {loss_meter.avg:.3f}, RTE: {rte_meter.avg:.3f}, RRE: {rre_meter.avg:.3f},",
            #         f"Hit Ratio: {hit_ratio_meter.avg:.3f}, Feat Match Ratio: {feat_match_ratio.avg:.3f}"
            #     ]))
            #     data_timer.reset()

        self.log.info(
            f'epoch {epoch}:lr:{lr[0]:.4f},loss:{loss_meter.avg:.3f},RTE:{rte_meter.avg:.3f},RRE:{rre_meter.avg:.3f},HitRatio:{hit_ratio_meter.avg:.3f},FeatMatchRatio:{feat_match_ratio.avg:.3f}')

        return {"loss": loss_meter.avg, "rre": rre_meter.avg, "rte": rte_meter.avg, 'feat_match_ratio': feat_match_ratio.avg, 'hit_ratio': hit_ratio_meter.avg}


class __backupHardestContrastiveLossTrainer(ContrastiveLossTrainer):

    def contrastive_hardest_negative_loss(self, F0, F1, positive_pairs, num_pos=5192, num_hn_samples=2048, thresh=None):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        N_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        if N_pos_pairs > num_pos:
            pos_sel = np.random.choice(N_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        # Find negatives for all F1[positive_pairs[:, 1]]
        subF0, subF1 = F0[sel0], F1[sel1]

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_loss = F.relu((posF0 - posF1).pow(2).sum(1) - self.pos_thresh)
        neg_loss0 = F.relu(self.neg_thresh - D01min[mask0]).pow(2)
        neg_loss1 = F.relu(self.neg_thresh - D10min[mask1]).pow(2)

        return pos_loss.mean(), (neg_loss0.mean() + neg_loss1.mean()) / 2

    def _train_epoch(self, epoch):
        gc.collect()
        self.model.train()
        # Epoch starts from 1
        total_loss = 0
        total_num = 0.0
        data_loader = self.data_loader
        data_loader_iter = self.data_loader.__iter__()
        iter_size = self.iter_size
        data_meter, data_timer, total_timer = timer.AverageMeter(), timer.Timer(), timer.Timer()
        start_iter = (epoch - 1) * (len(data_loader) // iter_size)
        for curr_iter in range(len(data_loader) // iter_size):
            self.optimizer.zero_grad()
            batch_pos_loss, batch_neg_loss, batch_loss = 0, 0, 0

            data_time = 0
            total_timer.tic()
            for iter_idx in range(iter_size):
                data_timer.tic()
                input_dict = data_loader_iter.next()
                data_time += data_timer.toc(average=False)

                sinput0 = ME.SparseTensor(input_dict['sinput0_F'].to(self.device), coordinates=input_dict['sinput0_C'].to(self.device))
                F0 = self.model(sinput0).F

                sinput1 = ME.SparseTensor(input_dict['sinput1_F'].to(self.device), coordinates=input_dict['sinput1_C'].to(self.device))

                F1 = self.model(sinput1).F

                pos_pairs = input_dict['correspondences']
                pos_loss, neg_loss = self.contrastive_hardest_negative_loss(
                    F0, F1, pos_pairs, num_pos=self.config.num_pos_per_batch * self.config.batch_size, num_hn_samples=self.config.num_hn_samples_per_batch * self.config.batch_size)

                pos_loss /= iter_size
                neg_loss /= iter_size
                loss = pos_loss + self.neg_weight * neg_loss
                loss.backward()

                batch_loss += loss.item()
                batch_pos_loss += pos_loss.item()
                batch_neg_loss += neg_loss.item()

            self.optimizer.step()
            gc.collect()

            torch.cuda.empty_cache()

            total_loss += batch_loss
            total_num += 1.0
            total_timer.toc()
            data_meter.update(data_time)

            if curr_iter % self.config.stat_every_iter == 0:
                self.writer.add_scalar('train/loss', batch_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/pos_loss', batch_pos_loss, start_iter + curr_iter)
                self.writer.add_scalar('train/neg_loss', batch_neg_loss, start_iter + curr_iter)
                self.log.info("Train Epoch: {} [{}/{}], Current Loss: {:.3e} Pos: {:.3f} Neg: {:.3f}".format(epoch, curr_iter,
                                                                                                             len(self.data_loader) // iter_size, batch_loss, batch_pos_loss, batch_neg_loss) +
                              "\tData time: {:.4f}, Train time: {:.4f}, Iter time: {:.4f}".format(data_meter.avg, total_timer.avg - data_meter.avg, total_timer.avg))
                data_meter.reset()
                total_timer.reset()


class __backupHardestTripletLossTrainer(TripletLossTrainer):

    def triplet_loss(self, F0, F1, positive_pairs, num_pos=1024, num_hn_samples=512, num_rand_triplet=1024):
        """
    Generate negative pairs
    """
        N0, N1 = len(F0), len(F1)
        num_pos_pairs = len(positive_pairs)
        hash_seed = max(N0, N1)

        # positive pairs downsample
        if num_pos_pairs > num_pos:
            pos_sel = np.random.choice(num_pos_pairs, num_pos, replace=False)
            sample_pos_pairs = positive_pairs[pos_sel]
        else:
            sample_pos_pairs = positive_pairs

        pos_ind0 = sample_pos_pairs[:, 0].long()
        pos_ind1 = sample_pos_pairs[:, 1].long()
        posF0, posF1 = F0[pos_ind0], F1[pos_ind1]

        # all downsample
        sel0 = np.random.choice(N0, min(N0, num_hn_samples), replace=False)  # just modify this line to accelerate training
        sel1 = np.random.choice(N1, min(N1, num_hn_samples), replace=False)

        subF0, subF1 = F0[sel0], F1[sel1]

        D01 = metrics.pdist(posF0, subF1, dist_type='L2')
        D10 = metrics.pdist(posF1, subF0, dist_type='L2')

        D01min, D01ind = D01.min(1)
        D10min, D10ind = D10.min(1)

        if not isinstance(positive_pairs, np.ndarray):
            positive_pairs = np.array(positive_pairs, dtype=np.int64)

        pos_keys = misc._hash(positive_pairs, hash_seed)

        D01ind = sel1[D01ind.cpu().numpy()]
        D10ind = sel0[D10ind.cpu().numpy()]
        neg_keys0 = misc._hash([pos_ind0.numpy(), D01ind], hash_seed)
        neg_keys1 = misc._hash([D10ind, pos_ind1.numpy()], hash_seed)

        mask0 = torch.from_numpy(np.logical_not(np.isin(neg_keys0, pos_keys, assume_unique=False)))
        mask1 = torch.from_numpy(np.logical_not(np.isin(neg_keys1, pos_keys, assume_unique=False)))
        pos_dist = torch.sqrt((posF0 - posF1).pow(2).sum(1) + 1e-7)

        # Random triplets
        rand_inds = np.random.choice(num_pos_pairs, min(num_pos_pairs, num_rand_triplet), replace=False)
        rand_pairs = positive_pairs[rand_inds]
        negatives = np.random.choice(N1, min(N1, num_rand_triplet), replace=False)

        # Remove positives from negatives
        rand_neg_keys = misc._hash([rand_pairs[:, 0], negatives], hash_seed)
        rand_mask = np.logical_not(np.isin(rand_neg_keys, pos_keys, assume_unique=False))
        anchors, positives = rand_pairs[torch.from_numpy(rand_mask)].T
        negatives = negatives[rand_mask]

        rand_pos_dist = torch.sqrt((F0[anchors] - F1[positives]).pow(2).sum(1) + 1e-7)
        rand_neg_dist = torch.sqrt((F0[anchors] - F1[negatives]).pow(2).sum(1) + 1e-7)

        loss = F.relu(torch.cat([rand_pos_dist + self.neg_thresh - rand_neg_dist, pos_dist[mask0] + self.neg_thresh - D01min[mask0], pos_dist[mask1] + self.neg_thresh - D10min[mask1]])).mean()

        return loss, pos_dist.mean(), (D01min.mean() + D10min.mean()).item() / 2
