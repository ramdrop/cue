import argparse
import os
from datetime import datetime

import gin
import pytorch_lightning as pl
import wandb
import torch.nn.functional as F
import MinkowskiEngine as ME

from src.models import get_model
from src.data import get_data_module
from src.modules import get_lightning_module
from src.utils.file import ensure_dir
from src.utils.logger import setup_logger
from src.utils.misc import logged_hparams, update_hparams
from src.utils.gin import get_all_gin_configurable_signatures, get_all_gin_parameters, get_gin_set_params




@gin.configurable
def train(project_name, run_name, save_path, lightning_module_name, data_module_name, model_name, gpus, log_every_n_steps, check_val_every_n_epoch, refresh_rate_per_second, best_metric, max_epoch,
          max_step):
    now = datetime.now().strftime('%m%d_%H%M%S')
    run_name = run_name + "_" + now
    save_path = os.path.join(save_path, run_name)
    ensure_dir(save_path)

    data_module = get_data_module(data_module_name)()
    model = get_model(model_name)()
    if model_name == 'Res16UNet34CSigma':   # We only train the sigma branch
        import torch
        def remove_prefix(k, prefix):
            return k[len(prefix):] if k.startswith(prefix) else k
        ckpt = torch.load('pretrained/res16unet34c_s3dis_4cm.ckpt')
        state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
        model.load_state_dict(state_dict, strict=False)
        for name, param in model.named_parameters():
            if 'sigma' in name:
                param.requires_grad = True
                print(f'{name} requires grad')
            else:
                param.requires_grad = False

    if model_name == 'Res16UNet34CMC':  # MC Dropout
        print(f'applying MC dropout with p={model.p} after every conv layer..')
        def dropout_hook_wrapper(module, sinput, soutput):
            input = soutput.F
            output = F.dropout(input, p=model.p, training=module.training)
            soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
            return soutput_new
        for module in model.modules():
            if isinstance(module, ME.MinkowskiConvolution):
                module.register_forward_hook(dropout_hook_wrapper)


    pl_module = get_lightning_module(lightning_module_name)(model=model, max_steps=max_step)
    gin.finalize()

    hparams = logged_hparams()
    # if 'Prob' in lightning_module_name or 'Aleatoric' in lightning_module_name:
    hparams = update_hparams(basekeys=hparams)

    callbacks = [
        pl.callbacks.TQDMProgressBar(refresh_rate=refresh_rate_per_second),
        pl.callbacks.ModelCheckpoint(dirpath=save_path, monitor=best_metric, save_last=True, save_top_k=1, mode="max"),
        pl.callbacks.LearningRateMonitor(),
    ]
    loggers = [
        pl.loggers.WandbLogger(
            name=run_name,
            save_dir=save_path,
            project=project_name,
            log_model=True,
            entity="ramdrop",          # set it to your wandb username
            config=hparams,
        )
    ]

    additional_kwargs = dict()
    # if gpus > 1:
    #     raise NotImplementedError("Currently, multi-gpu training is not supported.")

    trainer = pl.Trainer(
        default_root_dir=save_path,
        max_epochs=max_epoch,
        max_steps=max_step,
        accelerator='gpu',
        devices=gpus,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=log_every_n_steps,
        check_val_every_n_epoch=check_val_every_n_epoch,
        **additional_kwargs)

    # write config file
    with open(os.path.join(save_path, "config.gin"), "w") as f:
        f.write(gin.operative_config_str())

    wandb.save('src/models/*', policy="now")
    wandb.save('src/modules/segmentation.py', policy="now")
    wandb.save('src/data/*', policy="now")
    wandb.save('train.py', policy="now")

    trainer.fit(pl_module, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/scannet/train_res16unet34c.gin")              # train_res16unet34c_prob | train_res16unet34c | train_fpt.gin
    parser.add_argument("--seed", type=int, default=1235)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--debug", action="store_true")
    # parser.add_argument("-v", "--voxel_size", type=float, default=None)
    # parser.add_argument("--max_t", type=float, default=-1)
    # parser.add_argument("--metric_weight", type=float, default=-1)
    args = parser.parse_args()

    pl.seed_everything(args.seed)
    gin.parse_config_file(args.config)
    if args.debug:
        gin.bind_parameter("train.run_name", "debug")
    gin.bind_parameter("train.gpus", [args.gpus])

    # if args.voxel_size is not None:
    #     gin.bind_parameter("DimensionlessCoordinates.voxel_size", args.voxel_size)
    # gin.bind_parameter("Res16UNet34CProb.max_t", args.max_t)
    # gin.bind_parameter("LitSegMinkowskiModuleProb.metric_weight", args.metric_weight)

    setup_logger(gin.query_parameter('train.run_name'), args.debug, args.gpus)

    train()
