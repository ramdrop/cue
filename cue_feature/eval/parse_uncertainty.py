#%%
"""
Analysis
"""
import torch
import numpy as np
from os.path import join, exists, dirname
import os
import sys
import time
from tqdm import tqdm

DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)
sys.path.append('..')
from torch_cluster import knn
import matin
from glob import glob
import pickle
from utlis import uncertainty_util


tau1, tau2 = 0.01, 0.05
feature_dir = '/LOCAL2/ramdrop/github/point_registration/DPR/logs_modelnet/BTL_0522_095054/eval_val_0523_144457/features'

eval_dir = dirname(feature_dir)
hr_dir = join(eval_dir, 'hr')
if not exists(hr_dir):
    os.makedirs(hr_dir)
output_file = join(hr_dir, f"hr_uncertainty_{tau1*100:.0f}.pickle")

np.set_printoptions(precision=3)
print(f'tau1={tau1}, tau2={tau2}')

if not exists(output_file):
    print(f'{output_file} does not exist, processing ...')
    feature_files = glob(join(feature_dir, '*.npz'))
    num_sample = len(feature_files)
    bin_hit_ratios_pairs = np.zeros((num_sample, 10))
    bin_hit_ratios_pairs_counts = np.zeros((num_sample, 10))
    hit_ratios = np.zeros((num_sample, 1))
    for data_ind in tqdm(range(num_sample)):
        try:
            data_pack = np.load(join(feature_dir, f'{data_ind:0>5d}' + '.npz'))
        except:
            print(f'skip {data_ind}.npz')
            continue
        xyz, xyz_target, T_gt, feat, feat_target, sigma, sigma_target = \
            data_pack['xyz'], data_pack['xyz_target'], data_pack['T_gt'], data_pack['feat'], data_pack['feat_target'], data_pack['sigma'], data_pack['sigma_target']

        device = torch.device('cuda')
        xyz = torch.from_numpy(xyz).to(device)
        xyz_target = torch.from_numpy(xyz_target).to(device)
        T_gt = torch.from_numpy(T_gt).to(device)
        feat = torch.from_numpy(feat).to(device)
        feat_target = torch.from_numpy(feat_target).to(device)

        bin_hit_ratios_pair, bin_counts_pair, avg_hr = uncertainty_util.parse_hr_uncertainty(feat, feat_target, xyz, xyz_target, sigma, sigma_target, T_gt, tau1)
        bin_hit_ratios_pairs[data_ind] = bin_hit_ratios_pair
        bin_hit_ratios_pairs_counts[data_ind] = bin_counts_pair
        hit_ratios[data_ind] = avg_hr

    with open(output_file, 'wb') as handle:
        pickle.dump(bin_hit_ratios_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(bin_hit_ratios_pairs_counts, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(hit_ratios, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f'done.')
else:
    print(f'{output_file} exists, skip.')
