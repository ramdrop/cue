import os.path as osp
from typing import Optional

import gin
import numpy as np
import torch
import pytorch_lightning as pl

from src.data.scannet_loader import read_ply
from src.data.collate import CollationFunctionFactory
from src.data.sampler import InfSampler
import src.data.transforms as T

CLASSES = [
    'ceiling',                         # 0 - 28.9%
    'floor',                           # 1 - 35.4%
    'wall',                            # 2 - 29.2%
    'beam',                            # 3 - 0%
    'column',                          # 4 - 1.6%
    'window',                          # 5 - 3.8%
    'door',                            # 6 - 3.4%
    'chair',                           # 7 - 1.9%
    'table',                           # 8 - 3.8%
    'bookcase',                        # 9 - 11.5%
    'sofa',                            # 10 - 0.2%
    'board',                           # 11 - 1.3%
    'clutter',                         # 12 - 9%
]


def read_txt(path):
    """Read txt file into lines.
    """
    with open(path) as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    return lines


@gin.configurable
class S3DISArea5DatasetBase(torch.utils.data.Dataset):
    IN_CHANNELS = None
    NUM_CLASSES = 13
    SPLIT_FILES = {'train': ['area1.txt', 'area2.txt', 'area3.txt', 'area4.txt', 'area6.txt'], 'val': ['area5.txt'], 'test': ['area5.txt']}

    def __init__(self, phase, data_root, transform=None, ignore_label=255, ignore_class_labels=[]):
        assert self.IN_CHANNELS is not None
        assert phase in ['train', 'val', 'test']
        super(S3DISArea5DatasetBase, self).__init__()

        self.phase = phase
        self.data_root = data_root
        self.transform = transform
        self.ignore_label = ignore_label
        self.ignore_class_labels = ignore_class_labels
        self.labelmap = self.get_labelmap()

        self.split_files = self.SPLIT_FILES[phase]

        filenames = []
        for split_file in self.split_files:
            filenames += read_txt(osp.join(self.data_root, 'meta_data', split_file))
        self.filenames = [osp.join(self.data_root, 's3dis_processed', fname) for fname in filenames]

    def __len__(self):
        return len(self.filenames)

    def get_classnames(self):
        return CLASSES

    def __getitem__(self, idx):
        data = self._load_data(idx)
        coords, feats, labels = self.get_cfl_from_data(data)
        if self.transform is not None:
            coords, feats, labels = self.transform(coords, feats, labels)
        coords = torch.from_numpy(coords)
        feats = torch.from_numpy(feats)
        labels = torch.from_numpy(labels)
        return coords.float(), feats.float(), labels.long(), None

    def get_cfl_from_data(self, data):
        raise NotImplementedError

    def _load_data(self, idx):
        filename = self.filenames[idx]
        data = read_ply(filename)
        return data

    def get_labelmap(self):
        labelmap = {}
        new_label = 0
        for k in range(13):
            if k in self.ignore_class_labels:
                labelmap[k] = self.ignore_label
            else:
                labelmap[k] = new_label
                new_label += 1
        labelmap[255] = 255
        return labelmap


@gin.configurable
class S3DISArea5RGBDataset(S3DISArea5DatasetBase):
    IN_CHANNELS = 3

    def __init__(self, phase, data_root, transform=None, ignore_label=255):
        super(S3DISArea5RGBDataset, self).__init__(phase, data_root, transform, ignore_label)

    def get_cfl_from_data(self, data):
        xyz, rgb, label = data[:, :3], data[:, 3:6], data[:, 6]
        label = np.array([self.labelmap[x] for x in label])
        return (xyz.astype(np.float32), rgb.astype(np.float32), label.astype(np.int64))


@gin.configurable
class S3DISArea5RGBDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root,
        train_batch_size,
        val_batch_size,
        train_num_workers,
        val_num_workers,
        collation_type,
        train_transforms,
        eval_transforms,
    ):
        super(S3DISArea5RGBDataModule, self).__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.collate_fn = CollationFunctionFactory(collation_type)
        self.train_transforms_ = train_transforms
        self.eval_transforms_ = eval_transforms

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            train_transforms = []
            if self.train_transforms_ is not None:
                for name in self.train_transforms_:
                    train_transforms.append(getattr(T, name)())
            train_transforms = T.Compose(train_transforms)
            self.dset_train = S3DISArea5RGBDataset("train", self.data_root, train_transforms)
        eval_transforms = []
        if self.eval_transforms_ is not None:
            for name in self.eval_transforms_:
                eval_transforms.append(getattr(T, name)())
        eval_transforms = T.Compose(eval_transforms)
        self.dset_val = S3DISArea5RGBDataset("val", self.data_root, eval_transforms)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dset_train, batch_size=self.train_batch_size, sampler=InfSampler(self.dset_train, True), num_workers=self.train_num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, drop_last=False, collate_fn=self.collate_fn)
