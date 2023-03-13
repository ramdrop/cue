import os
import os.path
import sys
sys.path.append('.')

import numpy as np
import torch
import gin
import pytorch_lightning as pl
from typing import Optional
from src.data.collate import CollationFunctionFactory

label_name_mapping = {
    0: 'unlabeled',
    1: 'outlier',
    10: 'car',# keep
    11: 'bicycle',# keep
    13: 'bus',
    15: 'motorcycle',# keep
    16: 'on-rails',
    18: 'truck',# keep
    20: 'other-vehicle',# keep
    30: 'person',# keep
    31: 'bicyclist',# keep
    32: 'motorcyclist',# keep
    40: 'road',     # keep
    44: 'parking',# keep
    48: 'sidewalk',# keep
    49: 'other-ground',# keep
    50: 'building',# keep
    51: 'fence',# keep
    52: 'other-structure',
    60: 'lane-marking',
    70: 'vegetation',# keep
    71: 'trunk',# keep
    72: 'terrain',# keep
    80: 'pole',# keep
    81: 'traffic-sign',# keep
    99: 'other-object',
    252: 'moving-car',# keep
    253: 'moving-bicyclist',# keep
    254: 'moving-person',# keep
    255: 'moving-motorcyclist',# keep
    256: 'moving-on-rails',
    257: 'moving-bus',
    258: 'moving-truck',# keep
    259: 'moving-other-vehicle' # keep
}

kept_labels = [
    'road', 'sidewalk', 'parking', 'other-ground', 'building', 'car', 'truck',
    'bicycle', 'motorcycle', 'other-vehicle', 'vegetation', 'trunk', 'terrain',
    'person', 'bicyclist', 'motorcyclist', 'fence', 'pole', 'traffic-sign'
]

class SemanticKITTIDataset:

    def __init__(self, split, root, voxel_size, num_points, sample_stride=1):

        self.root = root
        self.split = split
        self.voxel_size = voxel_size
        self.sample_stride = sample_stride
        self.num_points = num_points
        self.seqs = []
        if split == 'train':
            self.seqs = [
                '00', '01', '02', '03', '04', '05', '06', '07', '09', '10'
            ]

        elif self.split == 'val':
            self.seqs = ['08']
        elif self.split == 'test':
            self.seqs = [
                '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21'
            ]

        self.files = []
        for seq in self.seqs:
            seq_files = sorted(os.listdir(os.path.join(self.root, seq, 'velodyne')))
            seq_files = [os.path.join(self.root, seq, 'velodyne', x) for x in seq_files]
            self.files.extend(seq_files)

        if self.sample_stride > 1:
            self.files = self.files[::self.sample_stride]

        reverse_label_name_mapping = {}
        self.label_map = np.zeros(260)
        cnt = 0
        for label_id in label_name_mapping:
            if label_id > 250:
                if label_name_mapping[label_id].replace('moving-', '') in kept_labels:
                    self.label_map[label_id] = reverse_label_name_mapping[label_name_mapping[label_id].replace('moving-', '')]
                else:
                    self.label_map[label_id] = 255
            elif label_id == 0:
                self.label_map[label_id] = 255
            else:
                if label_name_mapping[label_id] in kept_labels:
                    self.label_map[label_id] = cnt
                    reverse_label_name_mapping[label_name_mapping[label_id]] = cnt
                    cnt += 1
                else:
                    self.label_map[label_id] = 255

        self.reverse_label_name_mapping = reverse_label_name_mapping
        self.num_classes = cnt
        self.angle = 0.0

    def set_angle(self, angle):
        self.angle = angle

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        with open(self.files[index], 'rb') as b:
            block_ = np.fromfile(b, dtype=np.float32).reshape(-1, 4)
        block = np.zeros_like(block_)

        if 'train' in self.split:
            theta = np.random.uniform(0, 2 * np.pi)
            scale_factor = np.random.uniform(0.95, 1.05)
            rot_mat = np.array([[np.cos(theta), np.sin(theta), 0],
                                [-np.sin(theta),
                                 np.cos(theta), 0], [0, 0, 1]])

            block[:, :3] = np.dot(block_[:, :3], rot_mat) * scale_factor
        else:
            theta = self.angle
            transform_mat = np.array([[np.cos(theta),
                                       np.sin(theta), 0],
                                      [-np.sin(theta),
                                       np.cos(theta), 0], [0, 0, 1]])
            block[...] = block_[...]
            block[:, :3] = np.dot(block[:, :3], transform_mat)

        block[:, 3] = block_[:, 3]
        pc_ = np.round(block[:, :3] / self.voxel_size).astype(np.int32)
        pc_ -= pc_.min(0, keepdims=1)   # (124668, 3)

        label_file = self.files[index].replace('velodyne', 'labels').replace(
            '.bin', '.label')
        if os.path.exists(label_file):
            with open(label_file, 'rb') as a:
                all_labels = np.fromfile(a, dtype=np.uint32).reshape(-1) # (124668, )
        else:
            all_labels = np.zeros(pc_.shape[0]).astype(np.int32)

        labels_ = self.label_map[all_labels & 0xFFFF].astype(np.int64)  # semantic label in lower half, instance label in upper half, refer to https://github.com/PRBonn/semantic-kitti-api/blob/8e75f4d049b787321f68a11753cb5947b1e58e17/auxiliary/laserscan.py#L246
        # feat_ = np.ones((pc_.shape[0], 1))
        feat_ = block

        if 'train' in self.split:
            if feat_.shape[0] > self.num_points:
                inds = np.random.choice(np.arange(feat_.shape[0]), self.num_points, replace=False)  
                pc_ = pc_[inds]
                feat_ = feat_[inds]
                labels_ = labels_[inds]

        coords = torch.from_numpy(pc_)
        feats = torch.from_numpy(feat_)
        labels = torch.from_numpy(labels_)

        return coords.float(), feats.float(), labels.long(), None


@gin.configurable
class SemanticKITTIDataModule(pl.LightningDataModule):

    def __init__(
        self,
        data_root,
        train_batch_size,
        val_batch_size,
        train_num_workers,
        val_num_workers,
        collation_type,
        voxel_size,
        num_points,
        unlabeled_as_class=False,
    ):
        super(SemanticKITTIDataModule, self).__init__()
        self.data_root = data_root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.train_num_workers = train_num_workers
        self.val_num_workers = val_num_workers
        self.collate_fn = CollationFunctionFactory(collation_type)
        self.voxel_size = voxel_size
        self.num_points = num_points
        self.unlabeled_as_class = unlabeled_as_class

    def setup(self, stage: Optional[str] = None):
        if self.unlabeled_as_class:
            pass
        else:
            self.dset_train = SemanticKITTIDataset(split="train", root=self.data_root, voxel_size=self.voxel_size, sample_stride=1, num_points=self.num_points)

        if self.unlabeled_as_class:
            pass
        else:
            self.dset_val = SemanticKITTIDataset(split="val", root=self.data_root, voxel_size=self.voxel_size, sample_stride=1, num_points=self.num_points)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.dset_train, batch_size=self.train_batch_size, shuffle=True, drop_last=False, num_workers=self.train_num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.dset_val, batch_size=self.val_batch_size, shuffle=False, num_workers=self.val_num_workers, drop_last=False, collate_fn=self.collate_fn)

if __name__ == '__main__':
    train_dataset = SemanticKITTIDataset(split='train', root='/LOCAL2/ramdrop/github/point_registration/FastPointTransformer/dataset/semantic_kitti/sequences', voxel_size=0.02)
    for item in train_dataset[0]:
        if item is not None:
            print(item.shape)