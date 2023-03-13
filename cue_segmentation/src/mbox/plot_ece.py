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
from glob import glob
import pickle
import sys
sys.path.append('.')

import matin
from src.mbox import com

eval_dir_dic = com.get_eval_dic()
methods = ['entropy', 'aleatoric', 'mc', 'cue', 'lrmg']
labels = ['BL', 'BL+Aleatoric', 'BL+MCD', 'BL+Ours:DMG', 'BL+Ours:LRMG']
mask = [0, 1, 2, 3, 4]
methods = [methods[x] for x in mask]
labels = [labels[x] for x in mask]

plt.style.use('ggplot')
fig, axs = plt.subplots(2, 1, figsize=(5, 5 * 2), sharex=True, squeeze=False, dpi=200)
width = 0.015
offset = np.zeros(10) - 2 * width
for ind in range(len(methods)):
    method = methods[ind]
    label = labels[ind]
    with open(join(eval_dir_dic[method], 'ece', f'ece_{method}.pickle'), 'rb') as handle:
        precision_array = pickle.load(handle)
        count_array = pickle.load(handle)
        precision_avg_array = pickle.load(handle)
        ece_array = pickle.load(handle)
        print(f'Uncertainty method: {method}')
        print(f'CA - ece:  {ece_array.mean():.3f}+-{ece_array.var():.3f}')  # Calculate ECE then Average
        ece_e = com.cal_ece(precision_array.mean(axis=0), count_array.mean(axis=0)) # Average then Calculate ECE
        print(f'AC - ece:  {ece_e:.3f}+-{ece_e:.3f}')

    precision_array = precision_array.mean(0)
    density = count_array.mean(0)
    density = density/sum(density)

    ax = axs[0][0]
    ax.plot(np.arange(len(density))/10, precision_array, marker='o', markersize=5, linewidth=2, label=label)
    ax = axs[1][0]
    ax.bar(np.arange(len(density))/10+offset, density, width=0.02, label=labels[ind])

    offset += width

axs[0][0].plot(np.arange(len(density)) / 10, 1 - np.arange(len(density)) / 10, marker='o', markersize=0, linewidth=1, color='black', linestyle='--')
axs[0][0].set_ylabel('Precision')
axs[1][0].set_ylabel('Sample Density')
axs[1][0].set_xlabel('Uncertainty Level')
axs[1][0].set_xticks(np.arange(0, 1, 0.1))
axs[1][0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

matin.ax_default_style(axs[0][0], show_grid=True, show_legend=True, ratio=0.7)
matin.ax_default_style(axs[1][0], show_grid=True, show_legend=False, ratio=0.7)
plt.subplots_adjust(hspace=-0.35)
plt.savefig(join('results', 'ece_scannet.png'), bbox_inches='tight')
plt.savefig(join('results', 'ece_scannet.svg'))
