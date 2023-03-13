import gin
import numpy as np
import torch
import pytorch_lightning as pl
import pl_bolts
import torchmetrics
import warnings
import MinkowskiEngine as ME
import wandb

from src.utils.metric import per_class_iou
from src.losses.metric_loss import MetricLoss
from src.losses.aleatoric_loss import AleatoricLoss
from src.mbox import com

@gin.configurable
class LitSegmentationModuleBase(pl.LightningModule):

    def __init__(
        self,
        model,
        num_classes,
        lr,
        momentum,
        weight_decay,
        warmup_steps_ratio,
        max_steps,
        best_metric_type,
        ignore_label=255,
        dist_sync_metric=False,
        lr_eta_min=0.,
    ):
        super(LitSegmentationModuleBase, self).__init__()
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.best_metric_value = -np.inf if best_metric_type == "maximize" else np.inf
        self.metric = torchmetrics.ConfusionMatrix(num_classes=num_classes, compute_on_step=False, dist_sync_on_step=dist_sync_metric)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = pl_bolts.optimizers.LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=int(self.warmup_steps_ratio * self.max_steps),
            max_epochs=self.max_steps,
            eta_min=self.lr_eta_min,
        )
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}

    def training_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        logits = self.model(in_data)
        loss = self.criterion(logits, batch["labels"])
        self.log("train_loss_seg", loss.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)           # in_data:([744988, 3]), label:([744988])
        logits = self.model(in_data)    # ([744988, 13])
        loss = self.criterion(logits, batch["labels"])
        self.log("val_loss_seg", loss.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        pred = logits.argmax(dim=1, keepdim=False)
        mask = batch["labels"] != self.ignore_label
        self.metric(pred[mask], batch["labels"][mask])
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, outputs):
        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur

        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
        self.log("val_best_mIoU", self.best_metric_value, logger=True, prog_bar=True)
        self.log("val_mIoU", miou, logger=True, prog_bar=True)
        self.log("val_mAcc", macc, logger=True, prog_bar=True)

    def prepare_input_data(self, batch):
        raise NotImplementedError


@gin.configurable
class LitSegMinkowskiModule(LitSegmentationModuleBase):
    def prepare_input_data(self, batch):
        in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=self.model.QMODE)
        return in_data


@gin.configurable
class LitSegMinkowskiModuleRUL(LitSegMinkowskiModule):

    def __init__(self, model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label=255, dist_sync_metric=False, lr_eta_min=0):
        super().__init__(model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label, dist_sync_metric, lr_eta_min)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)

    def training_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        logits, index_dense, _ = self.model(in_data)                    # ([744988, 13])
        loss_segi = self.criterion(logits, batch["labels"])
        loss_segj = self.criterion(logits, batch["labels"][index_dense.flatten()])
        loss = (loss_segi + loss_segj)/2
        self.log("train_loss_segi", loss_segi.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss_segj", loss_segj.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)                               # in_data:([744988, 3]), label:([744988])
        logits, _, _ = self.model(in_data)                    # ([744988, 13])
        loss_seg = self.criterion(logits, batch["labels"])
        loss = loss_seg
        self.log("val_loss_seg", loss_seg.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        pred = logits.argmax(dim=1, keepdim=False)
        mask = batch["labels"] != self.ignore_label
        self.metric(pred[mask], batch["labels"][mask])
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, outputs):
        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur

        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
        self.log("val_best_mIoU", self.best_metric_value, logger=True, prog_bar=True)
        self.log("val_mIoU", miou, logger=True, prog_bar=True)
        self.log("val_mAcc", macc, logger=True, prog_bar=True)



@gin.configurable
class LitSegMinkowskiModuleDUL(LitSegMinkowskiModule):
    def __init__(self, model, lambda_kl, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label=255, dist_sync_metric=False, lr_eta_min=0):
        super().__init__(model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label, dist_sync_metric, lr_eta_min)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_label)
        self.lambda_kl = lambda_kl

    def kl(self, mu, logsigma2, log=False):
        mu = mu.F
        logsigma2 = logsigma2.F
        kl = -(1 + logsigma2 - mu.pow(2) - logsigma2.exp()) / 2 # ([N, 96])
        kl = kl.sum(dim=1).mean()
        return kl

    def training_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        logits, emb_mu, emb_logsigma2 = self.model(in_data)                    # ([Nr, num_of_cls]), ([N, feat_dim]), ([N, feat_dim])
        loss_seg = self.criterion(logits, batch["labels"])
        loss_kl = self.lambda_kl * self.kl(emb_mu, emb_logsigma2)
        loss = loss_seg + loss_kl
        self.log("train_loss_seg", loss_seg.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss_kl", loss_kl.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)           # in_data:([744988, 3]), label:([744988])
        logits, emb_mu, emb_logsigma2 = self.model(in_data)    # ([744988, 13])
        loss_seg = self.criterion(logits, batch["labels"])
        loss_kl = self.lambda_kl * self.kl(emb_mu, emb_logsigma2)
        loss = loss_seg + loss_kl
        self.log("val_loss_seg", loss_seg.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss_kl", loss_kl.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        pred = logits.argmax(dim=1, keepdim=False)
        mask = batch["labels"] != self.ignore_label
        self.metric(pred[mask], batch["labels"][mask])
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, outputs):
        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur

        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
        self.log("val_best_mIoU", self.best_metric_value, logger=True, prog_bar=True)
        self.log("val_mIoU", miou, logger=True, prog_bar=True)
        self.log("val_mAcc", macc, logger=True, prog_bar=True)

@gin.configurable
class LitSegMinkowskiModuleAleatoric(LitSegMinkowskiModule):
    def __init__(self, model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label=255, dist_sync_metric=False, lr_eta_min=0):
        super().__init__(model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label, dist_sync_metric, lr_eta_min)
        self.criterion = AleatoricLoss()

    def training_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        logits, sigma = self.model(in_data)                        # ([Nr, num_of_cls]), ([Nr, num_of_cls])
        loss = self.criterion(logits, sigma, batch["labels"])
        self.log("train_loss_seg", loss.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)           # in_data:([744988, 3]), label:([744988])
        logits, sigma = self.model(in_data)    # ([744988, 13])
        loss = self.criterion(logits, sigma, batch["labels"])
        self.log("val_loss_seg", loss.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        pred = logits.argmax(dim=1, keepdim=False)
        mask = batch["labels"] != self.ignore_label
        self.metric(pred[mask], batch["labels"][mask])
        torch.cuda.empty_cache()
        return loss

    def validation_epoch_end(self, outputs):
        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur

        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
        self.log("val_best_mIoU", self.best_metric_value, logger=True, prog_bar=True)
        self.log("val_mIoU", miou, logger=True, prog_bar=True)
        self.log("val_mAcc", macc, logger=True, prog_bar=True)

@gin.configurable
class LitSegMinkowskiModuleProb(LitSegMinkowskiModule):
    def __init__(self, model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, metric_weight, ignore_label=255, dist_sync_metric=False, lr_eta_min=0):
        super().__init__(model, num_classes, lr, momentum, weight_decay, warmup_steps_ratio, max_steps, best_metric_type, ignore_label, dist_sync_metric, lr_eta_min)
        self.metric_weight = metric_weight
        self.metric_criterion = MetricLoss()

    def training_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)
        logits, emb_mu, emb_sigma = self.model(in_data)                        # ([m, Nr, 13]), ([N, 96]), ([N, 1])

        loss_seg = self.criterion(logits.reshape(-1, logits.shape[-1]), batch["labels"][None].expand(logits.shape[0], *batch["labels"].shape).reshape(-1))
        if self.metric_weight > 0:
            emb_mu_dense = emb_mu.slice(in_data)    # TensorField
            emb_sigma_dense = emb_sigma.slice(in_data)     # TensorField
            xyz_dense = batch["coordinates"]
            label_dense = batch["labels"]

            xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
            labels_sparse = label_dense[unique_map]
            emb_mu_sparse = emb_mu_dense.F[unique_map]
            emb_sigma_sparse = emb_sigma_dense.F[unique_map]

            loss_metric, meta = self.metric_criterion(emb_mu_sparse, emb_sigma_sparse, xyz_sparse.to(self.device), labels_sparse.view(-1, 1)) # TODO could try (1) BTL (2) sysmetric KL (3) Wass.. ?
            loss = loss_seg + self.metric_weight * loss_metric
        else:
            loss_metric, meta = torch.tensor([0]).to(loss_seg), None
            loss = loss_seg
        self.log("train_loss_seg", loss_seg.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss_metric", loss_metric.item(), batch_size=batch["batch_size"], logger=True)
        self.log("train_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        if meta is not None:
            self.log("train_tri_probs", meta['probs'].item(), batch_size=batch["batch_size"], logger=True)
            self.log("train_tri_Tmu", meta['t_mu'].item(), batch_size=batch["batch_size"], logger=True)
            self.log("train_tri_Tsigma2", meta['t_sigma2'].item(), batch_size=batch["batch_size"], logger=True)
        torch.cuda.empty_cache()
        return loss

    def validation_step(self, batch, batch_idx):
        in_data = self.prepare_input_data(batch)           # in_data:([Nr, 3]), label:([Nr])
        logits, emb_mu, emb_sigma = self.model(in_data)    # ([Nr, 13])
        logits = logits.mean(dim=0)
        loss_seg = self.criterion(logits, batch["labels"])

        bin_precisions, bin_counts, precisions, ece_score, bin_precisions_ent, bin_counts_ent, precisions_ent, ece_score_ent = None, None, None, None, None, None, None, None
        if self.metric_weight > 0:
            if len(emb_mu) != len(in_data):
                emb_mu_dense = emb_mu.slice(in_data)           # TensorField
                emb_sigma_dense = emb_sigma.slice(in_data)     # TensorField
                xyz_dense = batch["coordinates"]
                label_dense = batch["labels"]

                xyz_sparse, unique_map = ME.utils.sparse_quantize(xyz_dense, return_index=True)
                labels_sparse = label_dense[unique_map]
                emb_mu_sparse = emb_mu_dense.F[unique_map]
                emb_sigma_sparse = emb_sigma_dense.F[unique_map]
            else:
                emb_mu_sparse = emb_mu
                emb_sigma_sparse = emb_sigma
                labels_sparse = batch["labels"]
                xyz_sparse = batch["coordinates"]

            loss_metric, meta = self.metric_criterion(emb_mu_sparse, emb_sigma_sparse, xyz_sparse.to(emb_mu_sparse.device), labels_sparse.view(-1, 1)) # TODO could try (1) BTL (2) sysmetric KL (3) Wass.. ?
            loss = loss_seg + self.metric_weight * loss_metric

            if self.current_epoch % 5 == 0:
                # ----------------- calculate uncertainty quality ---------------- #
                pred = logits.argmax(dim=1, keepdim=False)
                pred_sparse = pred[unique_map]
                bin_precisions, bin_counts, precisions = com.get_bins_precision(emb_sigma_sparse, pred_sparse, labels_sparse)
                ece_score = com.cal_ece(bin_precisions, bin_counts)
                ent = com.score_to_entropy(logits)
                ent_sparse = ent[unique_map]
                bin_precisions_ent, bin_counts_ent, precisions_ent = com.get_bins_precision(ent_sparse, pred_sparse, labels_sparse)
                ece_score_ent = com.cal_ece(bin_precisions_ent, bin_counts_ent)
        else:
            loss_metric, meta = torch.tensor([0]).to(loss_seg), None
            loss = loss_seg
        self.log("val_loss_seg", loss_seg.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss_metric", loss_metric.item(), batch_size=batch["batch_size"], logger=True)
        self.log("val_loss", loss.item(), batch_size=batch["batch_size"], logger=True)
        if meta is not None:
            self.log("train_tri_probs", meta['probs'].item(), batch_size=batch["batch_size"], logger=True)
            self.log("train_tri_Tmu", meta['t_mu'].item(), batch_size=batch["batch_size"], logger=True)
            self.log("train_tri_Tsigma2", meta['t_sigma2'].item(), batch_size=batch["batch_size"], logger=True)
        pred = logits.argmax(dim=1, keepdim=False)
        mask = batch["labels"] != self.ignore_label
        self.metric(pred[mask], batch["labels"][mask])
        torch.cuda.empty_cache()
        return [bin_precisions, bin_counts, precisions, ece_score, bin_precisions_ent, bin_counts_ent, precisions_ent, ece_score_ent]

    def validation_epoch_end(self, outputs):

        if self.metric_weight > 0 and self.current_epoch % 5 == 0:
            bin_precisions = np.array([x[0] for x in outputs]).mean(axis=0)
            bin_counts = np.array([x[1] for x in outputs]).mean(axis=0)
            precision = np.array([x[2] for x in outputs]).mean()
            ece_s = com.cal_ece(bin_precisions, bin_counts)
            ece_fig = com.vis_uncertainty_precision(bin_precisions, bin_counts, precision)
            self.logger.experiment.log({"ece_s": wandb.Image(ece_fig)}, step=self.global_step)

            bin_precisions = np.array([x[4] for x in outputs]).mean(axis=0)
            bin_counts = np.array([x[5] for x in outputs]).mean(axis=0)
            ece_e = com.cal_ece(bin_precisions, bin_counts)
            ece_fig = com.vis_uncertainty_precision(bin_precisions, bin_counts, precision)
            self.logger.experiment.log({"ece_e": wandb.Image(ece_fig)}, step=self.global_step)
            self.logger.experiment.log({"val_mECE_s": ece_s, "val_mECE_e": ece_e}, step=self.global_step)

        confusion_matrix = self.metric.compute().cpu().numpy()
        self.metric.reset()
        ious = per_class_iou(confusion_matrix) * 100
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            accs = confusion_matrix.diagonal() / confusion_matrix.sum(1) * 100
        miou = np.nanmean(ious)
        macc = np.nanmean(accs)

        def compare(prev, cur):
            return prev < cur if self.best_metric_type == "maximize" else prev > cur

        if compare(self.best_metric_value, miou):
            self.best_metric_value = miou
        self.log("val_best_mIoU", self.best_metric_value, logger=True, prog_bar=True)
        self.log("val_mIoU", miou, logger=True, prog_bar=True)
        self.log("val_mAcc", macc, logger=True, prog_bar=True)
