#%%
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from matplotlib import pyplot as plt
from os.path import exists, join, dirname
import os
import torch
import torch.nn.functional as F
import shutil
from scipy.special import softmax
from datetime import datetime
from tqdm import tqdm
import matin
from glob import glob
import pickle
import sys
sys.path.append('.')
from src.mbox import com

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--uncertainty_method", type=str, default='rul')           # cue | lrmg | entropy | aleatoric | mc | dul | rul
args = parser.parse_args()

eval_dir_dic = com.get_eval_dic()
eval_dir = eval_dir_dic[args.uncertainty_method]
print(f'evaluating {args.uncertainty_method}: {eval_dir} ..')

FILE_IND = 0
filelist = glob(join(eval_dir, 'meta', '*_label_dense.npy'))
pcd_dir = join(eval_dir, 'pcd')
if not exists(pcd_dir):
    os.makedirs(pcd_dir)
ece_dir = join(eval_dir, 'ece')
if not exists(ece_dir):
    os.makedirs(ece_dir)

filelist.sort()

ece_array = np.zeros((len(filelist)))
precision_array, count_array, precision_avg_array = np.zeros((len(filelist), 10)), np.zeros((len(filelist), 10)), np.zeros((len(filelist)))
for ind in tqdm(range(len(filelist))):
    label_file = filelist[ind]
    label, sigma, seg_logit, pred = com.load_meta(label_file, 'label', 'sigma', 'seg_logit', 'pred')
    if args.uncertainty_method == 'cue':
        pass
    elif args.uncertainty_method == 'entropy':
        sigma = com.score_to_entropy(seg_logit)
    elif args.uncertainty_method == 'aleatoric':
        y = F.one_hot(torch.from_numpy(pred), num_classes=20).numpy() # [Nr, 20]
        q = (sigma - sigma.min(axis=1, keepdims=True)) / (sigma.max(axis=1, keepdims=True) - sigma.min(axis=1, keepdims=True))# [Nr, 20]
        sigma = y * (1 - 0.5 * q) + (1 - y) * 0.5 * q
        sigma = sigma.sum(axis=1)
    elif args.uncertainty_method in ['mc', 'mc_01', 'mc_005' ]:
        # sigma = com.score_to_entropy(seg_logit)
        y = F.one_hot(torch.from_numpy(pred), num_classes=20).numpy() # [Nr, 20]
        q = (sigma - sigma.min(axis=1, keepdims=True)) / (sigma.max(axis=1, keepdims=True) - sigma.min(axis=1, keepdims=True))# [Nr, 20]
        sigma = y * (1 - 0.5 * q) + (1 - y) * 0.5 * q
        sigma = sigma.sum(axis=1)
    elif args.uncertainty_method in ['dul', 'rul']:
        sigma = np.exp(sigma)
    else:
        raise Exception('undefined uncertainty method')
    precisions_, counts_, precision_avg_ = com.get_bins_precision(sigma, pred, label, prune_ignore=False)
    ece_array[ind] = com.cal_ece(precisions_, counts_)
    # com.vis_uncertainty_precision(bin_precisions, bin_counts, precision, join(ece_dir, f"sigma_{label_file.split('/')[-1].split('.')[0]}.png"))
    precision_array[ind] = (precisions_)
    count_array[ind] = (counts_)
    precision_avg_array[ind] = (precision_avg_)


print(f'Uncertainty method: {args.uncertainty_method}')
print(f'ece:  {ece_array.mean():.3f}+-{ece_array.var():.3f}')  # Calculate ECE then Average

with open(join(eval_dir, 'ece', f'ece_{args.uncertainty_method}.pickle'), 'wb') as handle:
    pickle.dump(precision_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(count_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(precision_avg_array, handle, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(ece_array, handle, protocol=pickle.HIGHEST_PROTOCOL)


fig, axs = plt.subplots(1, 1, figsize=(5, 6), squeeze=False, dpi=200)
ax = axs[0][0]
ax.plot(np.arange(precision_array.shape[1]), np.array(precision_array).mean(axis=0), marker='o', label=f'{args.uncertainty_method}')
ax.plot(np.arange(precision_array.shape[1]), 1-np.arange((precision_array.shape[1]))/10.0, lw=1, color='black')
ax.set_xlabel('Uncertainty')
ax.set_ylabel('Precision')
matin.ax_default_style(ax, show_grid=True, show_legend=True)
ax.set_ylim([0, 1])
matin.ax_lims(ax, interval_xticks=1)
plt.savefig(join(eval_dir, 'ece', 'ece.png'), bbox_inches='tight')
