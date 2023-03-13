from enum import unique
import gc
import argparse
from posixpath import dirname
from os.path import join, exists
import os
from datetime import datetime
import pandas as pd
import copy
from tqdm import tqdm
import gin
import torch
import torch.nn.functional as F
import torchmetrics
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import numpy as np
import wandb
from rich.console import Console
from rich.progress import track
from rich.table import Table

from src.models import get_model
from src.data import get_data_module
from src.utils.metric import per_class_iou
import src.data.transforms as T
from src.mbox import com
np.set_printoptions(precision=3, suppress=True)

def print_results(classnames, confusion_matrix):    # (21, 21)
    # results
    ious = per_class_iou(confusion_matrix) * 100    # (21,)
    accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100         # (21,)
    miou = np.nanmean(ious)
    macc = np.nanmean(accs)
    print(f'miou:{miou:.4f}')
    print(f'macc:{macc:.4f}')
    print(f'ious:{ious}')
    df = pd.DataFrame(data=ious)
    df.to_csv('baseline.csv', header=0, float_format='%.4f')

    # print results
    console = Console()
    table = Table(show_header=True, header_style="bold")

    columns = ["mAcc", "mIoU"]
    num_classes = len(classnames)
    for i in range(num_classes):
        columns.append(classnames[i])
    for col in columns:
        table.add_column(col)
    ious = ious.tolist()
    row = [macc, miou, *ious]
    table.add_row(*[f"{x:.2f}" for x in row])
    # console.print(table)


def get_rotation_matrices(num_rotations=8):
    angles = [2 * np.pi / num_rotations * i for i in range(num_rotations)]
    rot_matrices = []
    for angle in angles:
        rot_matrices.append(torch.Tensor([[np.cos(angle), -np.sin(angle), 0, 0], [np.sin(angle), np.cos(angle), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
    return rot_matrices


@torch.no_grad()
def infer(model, batch, device):
    in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
    pred = model(in_data).cpu()
    return pred
    # if type(pred) is tuple:
    #     pred = pred[0].squeeze(0).cpu()
    # else:
    #     pred = pred.cpu()
    # return pred


@torch.no_grad()
def infer_with_rotation_average(model, batch, device):
    rotation_matrices = get_rotation_matrices()
    pred = torch.zeros((len(batch["labels"]), model.out_channels), dtype=torch.float32)
    for M in rotation_matrices:
        batch_, coords_ = torch.split(batch["coordinates"], [1, 3], dim=1)
        coords = T.homogeneous_coords(coords_) @ M
        coords = torch.cat([batch_, coords[:, :3].float()], dim=1)

        in_data = ME.TensorField(features=batch["features"], coordinates=coords, quantization_mode=model.QMODE, device=device)
        pred += model(in_data).cpu()

        gc.collect()
        torch.cuda.empty_cache()

    # pred = pred.argmax(dim=1)
    return pred


@gin.configurable
def eval(
    checkpoint_path,
    model_name,
    data_module_name,
    use_rotation_average
):

    eval_dir = join(dirname(checkpoint_path), f"eval_{datetime.now().strftime('%m%d_%H%M%S')}")
    meta_dir = join(eval_dir, 'meta')
    if not exists(meta_dir):
        os.makedirs(meta_dir)

    assert torch.cuda.is_available()
    torch.cuda.set_device(1)
    device = torch.device("cuda")

    # file = wandb.restore('src/models/resunet.py', run_path='ramdrop/FastPointTransformer-release/ahuvo7vm')

    ckpt = torch.load(checkpoint_path)

    def remove_prefix(k, prefix):
        return k[len(prefix):] if k.startswith(prefix) else k

    state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
    model = get_model(model_name)()

    if 'MC' in model_name:
        p_mc = gin.query_parameter('Res16UNet34CMC.p')
        print(f'applying MC dropout p={p_mc} after every conv layer..')
        def dropout_hook_wrapper_another(module, sinput, soutput):
            soutput = MEF.dropout(soutput, p=0.2, training=module.training)
            return soutput
        def dropout_hook_wrapper(module, sinput, soutput):
            input = soutput.F
            output = F.dropout(input, p=p_mc, training=module.training)
            soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
            return soutput_new
        for module in model.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                module.register_forward_hook(dropout_hook_wrapper)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    data_module = get_data_module(data_module_name)()
    data_module.setup("test")
    val_loader = data_module.val_dataloader()

    confmat = torchmetrics.ConfusionMatrix(num_classes=data_module.dset_val.NUM_CLASSES, compute_on_step=False)

    outputs = []
    ignore_portion = []
    if 'Prob' in model_name:
        print('CUE eval route')
        with torch.inference_mode(mode=True):
            for index, batch in enumerate(track(val_loader)):
                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                logits, emb_mu, emb_sigma = model(in_data)                     # ([Nr, 13])
                logits = logits.mean(dim=0)
                pred_dense = logits.argmax(dim=1, keepdim=False)
                emb_mu_dense = emb_mu.slice(in_data)           # TensorField
                emb_sigma_dense = emb_sigma.slice(in_data)     # TensorField
                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]

                xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
                labels_sparse = label_dense[unique_map]
                emb_mu_sparse = emb_mu_dense.F[unique_map]
                emb_sigma_sparse = emb_sigma_dense.F[unique_map]
                logits_sparse = logits[unique_map]   # logits[:,unique_map,:]
                rgb_sparse = batch["features"][unique_map]
                pred_sparse = pred_dense[unique_map]
                bin_precisions, bin_counts, precisions = com.get_bins_precision(emb_sigma_sparse, pred_sparse, labels_sparse)
                outputs.append([bin_precisions, bin_counts, precisions])
                np.save(join(meta_dir, f'{index}_seg_logit.npy'), logits_sparse.cpu().numpy())     # ([m, N, 13])
                np.save(join(meta_dir, f'{index}_pred.npy'), pred_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma.npy'), emb_sigma_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz.npy'), xyz_sparse[:,1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb.npy'), rgb_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_label.npy'), labels_sparse.cpu().numpy())

                np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())     # ([m, Nr, 13])
                np.save(join(meta_dir, f'{index}_pred_dense.npy'), pred_dense.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma_dense.npy'), emb_sigma_dense.F.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz_dense.npy'), xyz_dense[:,1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())
                np.save(join(meta_dir, f'{index}_label_dense.npy'), label_dense.cpu().numpy())

                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1).cpu()          # mean(dim=0).
                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()
    elif 'DUL' in model_name:
        print('DUL eval route')
        with torch.inference_mode(mode=True):
            for index, batch in enumerate(track(val_loader)):
                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                logits, emb_mu, emb_sigma = model(in_data)                     # ([Nr, 13]) out_logit, out_mu, out_logsigma2
                pred_dense = logits.argmax(dim=1, keepdim=False)
                emb_mu_dense = emb_mu.slice(in_data)                           # TensorField
                emb_sigma_dense = emb_sigma.slice(in_data)                     # TensorField
                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]

                xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
                labels_sparse = label_dense[unique_map]
                emb_mu_sparse = emb_mu_dense.F[unique_map]
                emb_sigma_sparse = emb_sigma_dense.F[unique_map]
                logits_sparse = logits[unique_map]                                                 # logits[:,unique_map,:]
                rgb_sparse = batch["features"][unique_map]
                pred_sparse = pred_dense[unique_map]
                bin_precisions, bin_counts, precisions = com.get_bins_precision(emb_sigma_sparse, pred_sparse, labels_sparse)
                outputs.append([bin_precisions, bin_counts, precisions])
                np.save(join(meta_dir, f'{index}_seg_logit.npy'), logits_sparse.cpu().numpy())     # ([m, N, 13])
                np.save(join(meta_dir, f'{index}_pred.npy'), pred_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma.npy'), emb_sigma_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz.npy'), xyz_sparse[:, 1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb.npy'), rgb_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_label.npy'), labels_sparse.cpu().numpy())

                np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())      # ([m, Nr, 13])
                np.save(join(meta_dir, f'{index}_pred_dense.npy'), pred_dense.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma_dense.npy'), emb_sigma_dense.F.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz_dense.npy'), xyz_dense[:, 1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())
                np.save(join(meta_dir, f'{index}_label_dense.npy'), label_dense.cpu().numpy())

                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1).cpu()          # mean(dim=0).
                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()
    elif 'RUL' in model_name:
        print('RUL eval route')
        with torch.inference_mode(mode=True):
            for index, batch in enumerate(track(val_loader)):
                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                logits_sparse, emb_mu, emb_sigma_sparse = model(in_data)                     # ([Nr, 13]) out_logit, out_mu, out_logsigma2

                logits = logits_sparse.slice(in_data).F
                pred_dense = logits.argmax(dim=1, keepdim=False)
                emb_sigma_dense = emb_sigma_sparse.slice(in_data).F

                logits_sparse = logits_sparse.F
                pred_sparse = logits_sparse.argmax(dim=1, keepdim=False)
                emb_sigma_sparse = emb_sigma_sparse.F

                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]
                xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
                labels_sparse = label_dense[unique_map]
                rgb_sparse = batch["features"][unique_map]

                bin_precisions, bin_counts, precisions = com.get_bins_precision(emb_sigma_sparse, pred_sparse, labels_sparse)
                outputs.append([bin_precisions, bin_counts, precisions])

                np.save(join(meta_dir, f'{index}_seg_logit.npy'), logits_sparse.cpu().numpy())     # ([m, N, 13])
                np.save(join(meta_dir, f'{index}_pred.npy'), pred_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma.npy'), emb_sigma_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz.npy'), xyz_sparse[:, 1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb.npy'), rgb_sparse.cpu().numpy())
                np.save(join(meta_dir, f'{index}_label.npy'), labels_sparse.cpu().numpy())

                np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())      # ([m, Nr, 13])
                np.save(join(meta_dir, f'{index}_pred_dense.npy'), pred_dense.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma_dense.npy'), emb_sigma_dense.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz_dense.npy'), xyz_dense[:, 1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())
                np.save(join(meta_dir, f'{index}_label_dense.npy'), label_dense.cpu().numpy())

                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1).cpu()          # mean(dim=0).
                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()
    elif 'Aleatoric' in model_name:
        print('Aleatoric eval route')
        with torch.inference_mode(mode=True):
            for index, batch in enumerate(track(val_loader)):
                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                logits, sigma = model(in_data)    # ([Nr, num_cls])
                pred_dense = logits.argmax(dim=1, keepdim=False)
                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]

                np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())     # ([m, Nr, 13])
                np.save(join(meta_dir, f'{index}_pred_dense.npy'), pred_dense.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma_dense.npy'), sigma.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz_dense.npy'), xyz_dense[:,1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())
                np.save(join(meta_dir, f'{index}_label_dense.npy'), label_dense.cpu().numpy())

                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1).cpu()          # mean(dim=0).
                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()
    elif 'MC' in model_name:
        print('MC eval route')
        num_repeat = 40
        model.train()
        # for module in model.modules():
        #     if isinstance(module, ME.MinkowskiConvolution):
        #         module.training = True
        with torch.no_grad():
            for index, batch in enumerate(track(val_loader)):
                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                logits_multiple = torch.zeros((num_repeat, len(in_data), 20)).to(device)
                for ind in range(num_repeat):
                    # in_data_ = copy.deepcopy(in_data)
                    in_data_ = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                    logits_multiple[ind] = model(in_data_) # ([Nr, num_cls])
                sigma, logits = torch.var_mean(logits_multiple, dim=0)

                pred_dense = logits.argmax(dim=1, keepdim=False)
                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]

                np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())     # ([m, Nr, 13])
                np.save(join(meta_dir, f'{index}_pred_dense.npy'), pred_dense.cpu().numpy())
                np.save(join(meta_dir, f'{index}_sigma_dense.npy'), sigma.cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz_dense.npy'), xyz_dense[:,1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())
                np.save(join(meta_dir, f'{index}_label_dense.npy'), label_dense.cpu().numpy())

                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1).cpu()          # mean(dim=0).
                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()
        pass
    else:
        print('default eval route')
        infer_fn = infer_with_rotation_average if use_rotation_average else infer
        with torch.inference_mode(mode=True):
            for index, batch in enumerate(track(val_loader)):
                logits = infer_fn(model, batch, device)
                mask = batch["labels"] != data_module.dset_val.ignore_label
                ignore_portion.append(mask.float().mean())
                pred = logits.argmax(dim=1)

                np.save(join(meta_dir, f'{index}_seg_logit_dense.npy'), logits.cpu().numpy())     # ([m, Nr, 13])
                np.save(join(meta_dir, f'{index}_pred_dense.npy'), logits.argmax(dim=1, keepdim=False).cpu().numpy())
                np.save(join(meta_dir, f'{index}_label_dense.npy'), batch["labels"].cpu().numpy())
                np.save(join(meta_dir, f'{index}_xyz_dense.npy'), batch["coordinates"][:,1:].cpu().numpy())
                np.save(join(meta_dir, f'{index}_rgb_dense.npy'), batch["features"].cpu().numpy())

                confmat(pred[mask], batch["labels"][mask])
                torch.cuda.empty_cache()

    confmat = confmat.compute().numpy() # (21, 21)
    cnames = data_module.dset_val.get_classnames()
    print_results(cnames, confmat)

    if 'Prob' in model_name:
        bin_precisions = np.array([x[0] for x in outputs]).mean(axis=0)
        bin_counts = np.array([x[1] for x in outputs]).mean(axis=0)
        precision = np.array([x[2] for x in outputs]).mean()
        ece_s = com.cal_ece(bin_precisions, bin_counts)
        print(f'ece_s:{ece_s:.3f}')

    ignore_portion = np.array(ignore_portion)
    print(f'labeled points/all points:{ignore_portion.mean():.3f}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/s3dis/eval_res16unet34c.gin")
    parser.add_argument("--ckpt_path", type=str, default="./pretrained/res16unet34c_s3dis_4cm.ckpt") # |
    parser.add_argument("-r", "--use_rotation_average", action="store_true")
    parser.add_argument("-v", "--voxel_size", type=float, default=None)        # overwrite voxel_size
    args = parser.parse_args()

    gin.parse_config_file(args.config)
    if args.voxel_size is not None:
        gin.bind_parameter("DimensionlessCoordinates.voxel_size", args.voxel_size)

    eval(args.ckpt_path, use_rotation_average=args.use_rotation_average)