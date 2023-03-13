#%%
from ast import Raise
from logging import raiseExceptions
import os
from os.path import exists, join, dirname
import sys
sys.path.append('..')
from glob import glob
import numpy as np
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import open3d as o3d
import colorer
from tqdm import tqdm
import com
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--uncertainty_method", type=str, default='rul')                   # cue | lrmg | entropy | aleatoric | mc | dul | rul
parser.add_argument("-i", "--id", type=int, default=-1)
args = parser.parse_args()
print(f'=> uncertainty method: {args.uncertainty_method}')

def render_color(feat, type='rgb', color_map=None):
    if isinstance(feat, int):                    # feat==-1
        return -1
    if type in ['rgb']:
        color = feat
        if color.mean() > 1:
            color = color / 255.0
    elif type in ['label', 'pred']:
        color = np.zeros((feat.shape[0],3))
        for k in color_map.keys():
            color[feat==k] = color_map[k]
    elif type in ['sigma']:
        feat_normalized = (feat - np.min(feat)) / (np.max(feat) - np.min(feat))
        feat_normalized = feat_normalized.flatten()
        color = plt.cm.viridis(feat_normalized)[:,:3]
    else:
        raise Exception('undefined feat')
    return color

eval_dir_dic = com.get_eval_dic()
eval_dir = eval_dir_dic[args.uncertainty_method]
print(f'evaluating {args.uncertainty_method}: {eval_dir} ..')

pcd_dir = join(eval_dir, 'pcd')
if not exists(pcd_dir):
    os.makedirs(pcd_dir)
color_map = colorer.get_color_map('scannet')
filelist = glob(join(eval_dir, 'meta', '*_label_dense.npy'))
filelist.sort()

# FILE_IND = args.id if args.id > 0 else 0
# label_file = filelist[FILE_IND]

for ind in tqdm(range(len(filelist))):
    label_file = filelist[ind]

    xyz, rgb, label, sigma, seg_logit, pred = com.load_meta(label_file, 'xyz', 'rgb', 'label', 'sigma', 'seg_logit', 'pred')
    # print('=> loaded meta.')

    # REDUCE SIGMA ======================= #
    if args.uncertainty_method in ['cue', 'lrmg']:
        pass
    elif args.uncertainty_method == 'entropy':
        sigma = com.score_to_entropy(seg_logit)
    elif args.uncertainty_method == 'aleatoric':
        y = F.one_hot(torch.from_numpy(pred), num_classes=20).numpy() # [Nr, 20]
        q = (sigma - sigma.min(axis=1, keepdims=True)) / (sigma.max(axis=1, keepdims=True) - sigma.min(axis=1, keepdims=True))# [Nr, 20]
        sigma = y * (1 - 0.5 * q) + (1 - y) * 0.5 * q
        sigma = sigma.sum(axis=1)
    elif args.uncertainty_method in ['mc', 'mc_01', 'mc_005']:
        y = F.one_hot(torch.from_numpy(pred), num_classes=20).numpy() # [Nr, 20]
        q = (sigma - sigma.min(axis=1, keepdims=True)) / (sigma.max(axis=1, keepdims=True) - sigma.min(axis=1, keepdims=True))# [Nr, 20]
        sigma = y * (1 - 0.5 * q) + (1 - y) * 0.5 * q
        sigma = sigma.sum(axis=1)
    elif args.uncertainty_method in ['dul', 'rul']:
        sigma = np.exp(sigma)
    else:
        raise Exception('Wrong uncertainty type.')
    # print('=> reduced sigma.')

    # RESTORE LABEL ====================== #
    label = colorer.restore_label(label)
    pred = colorer.restore_label(pred)

    # RENDER COLOR ======================= #
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    for feat, type in zip([rgb, label, pred, sigma], ['rgb', 'label', 'pred', 'sigma']):
        color = render_color(feat, type, color_map)
        pcd.colors = o3d.utility.Vector3dVector(color)
        o3d.io.write_point_cloud(f"{label_file.replace('label', type).replace('npy', 'ply').replace('meta', 'pcd')}", pcd)
        # print(f"=> write {label_file.replace('label', type).replace('npy', 'ply').replace('meta', 'pcd')}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    color = np.zeros((rgb.shape[0],3))
    for k in color_map.keys():
        color[label == k] = color_map[k]
    color[label == pred] = colorer.RGB(0, 0, 0)
    color[label != pred] = colorer.RGB(214, 39, 40)
    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud(f"{label_file.replace('label', 'error').replace('npy', 'ply').replace('meta', 'pcd')}", pcd)
