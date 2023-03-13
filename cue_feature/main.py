#%%
import hydra
from omegaconf import DictConfig, OmegaConf

import dataset_scannet
import trainers

TRAINER_LIST = []
TRAINER_LIST.extend([getattr(trainers, cls_name) for cls_name in dir(trainers) if 'Trainer' in cls_name])
TRAINER_LUT = {t.__name__: t for t in TRAINER_LIST}

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    cfg = cfg.train

    Trainer = TRAINER_LUT[f'{cfg.trainer}Trainer']
    if cfg.dataset in ['ThreeDMatchPairDataset']:
        train_dataloader = dataset_scannet.make_data_loader(cfg, 'train', cfg.batch_size, num_threads=cfg.train_num_thread)
        val_dataloader = dataset_scannet.make_data_loader(cfg, 'val', cfg.val_batch_size, num_threads=cfg.val_num_thread)

    trainer = Trainer(config=cfg, train_dataloader=train_dataloader, val_dataloader=val_dataloader)
    trainer.train()

if __name__ == "__main__":
    main()
