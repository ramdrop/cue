#%%
import numpy as np

np.set_printoptions(precision=3, suppress=True)
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
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

print('=> start processing...')

eval_dir_dic = com.get_eval_dic()
methods = ['entropy', 'aleatoric', 'dul', 'rul', 'mc', 'mc_01', 'mc_005', 'cue', 'lrmg']
labels = ['Mink+SE', 'Mink+AU', 'Mink+DUL','Mink+RUL','Mink+MCD(p=0.2)', 'Mink+MCD(p=0.1)', 'Mink+MCD(p=0.05)', 'Mink+CUE', 'Mink+CUE+']
colors = ['#168aad', '#498467', '#3A4F7A', '#B08BBB', '#D2D79F', '#483838', '#8ecae6', '#FFC4C4', '#B25068']

mask = [0, 1, 2, 3, 6, 7, 8]
methods = [methods[x] for x in mask]
labels = [labels[x] for x in mask]
colors = [colors[x] for x in mask]

plt.style.use('ggplot')
# fig, axs = plt.subplots(3, 1, figsize=(5, 5 * 3), sharex=True, squeeze=False, dpi=200)
plt.figure(facecolor='white', figsize=(5, 5 * 3), dpi=200)
axs = [[1],[2], [3]]
axs[0][0] = plt.axes([0.2, 0.175, 0.8, 0.4])
axs[1][0] = plt.axes([0.2, 0.055, 0.8, 0.4])
axs[2][0] = plt.axes([0.2, 0, 0.8, 0.4])
axs[1][0].get_shared_x_axes().join(axs[1][0], axs[0][0])
axs[1][0].get_shared_x_axes().join(axs[1][0], axs[2][0])

width = 0.01
offset = np.zeros(10) - 2 * width
for ind in range(len(methods)):
    method = methods[ind]
    label = labels[ind]
    with open(join(eval_dir_dic[method], 'ece', f'ece_{method}.pickle'), 'rb') as handle:
        precision_array = pickle.load(handle)
        count_array = pickle.load(handle)
        precision_avg_array = pickle.load(handle)
        ece_array = pickle.load(handle)
        # print(f'Uncertainty method: {method}')
        # print(f'CA - ece:  {ece_array.mean():.3f}+-{ece_array.var():.3f}')  # Calculate ECE then Average
        ece_e = com.cal_ece(precision_array.mean(axis=0), count_array.mean(axis=0)) # Average then Calculate ECE
        # print(f'AC - ece:  {ece_e:.3f}+-{ece_e:.3f}')

    precision_array = precision_array.mean(0)
    density = count_array.mean(0)
    density = density/sum(density)

    if ind == 0:
        axs[0][0].plot(np.arange(len(density)) / 10, 1 - np.arange(len(density)) / 10, marker='o', markersize=0, linewidth=1, color='black', linestyle='--', label='ideal', alpha=0.5)

    axs[0][0].plot(np.arange(len(density)) / 10, precision_array, marker='o', markersize=3, linewidth=2, label=labels[ind], color=colors[ind])
    axs[1][0].bar(np.arange(len(density))/10+offset, density, width=width, label=labels[ind], color=colors[ind])
    axs[2][0].bar(np.arange(len(density))/10+offset, density, width=width, label=labels[ind], color=colors[ind])

    offset += width

axs[0][0].set_ylabel('Precision')
axs[2][0].set_ylabel('Sample Density')
axs[2][0].yaxis.set_label_coords(-0.08, 0.70)
axs[2][0].set_xlabel('Uncertainty Level')
axs[1][0].set_xticks(np.arange(0, 1, 0.1))
axs[1][0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
axs[0][0].set_xticks(np.arange(0, 1, 0.1))
axs[0][0].set_xticklabels([])
axs[2][0].set_xticks(np.arange(0, 1, 0.1))
axs[2][0].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])

matin.ax_default_style(axs[0][0], show_grid=True, show_legend=True, ratio=0.7)
matin.ax_default_style(axs[1][0], show_grid=False, show_legend=False, ratio=0.7)
matin.ax_default_style(axs[2][0], show_grid=False, show_legend=False, ratio=0.7)

axs[2][0].yaxis.set_major_locator(MultipleLocator(0.1))
axs[1][0].yaxis.set_major_locator(MultipleLocator(0.1))

axs[1][0].set_ylim([0.65, 0.75])
axs[2][0].set_ylim([0, 0.28])
axs[1][0].spines.bottom.set_visible(False)
axs[2][0].spines.top.set_visible(False)
axs[1][0].xaxis.tick_top()
axs[1][0].tick_params(labeltop=False)  # don't put tick labels at the top
axs[2][0].xaxis.tick_bottom()

d = .2  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=6,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
axs[1][0].plot([0, 1], [0, 0], transform=axs[1][0].transAxes, **kwargs)
axs[2][0].plot([0, 1], [1, 1], transform=axs[2][0].transAxes, **kwargs)

# fig.subplots_adjust(hspace=-0.5)
plt.savefig(join('results', 'ece_scannet.png'), bbox_inches='tight')
plt.savefig(join('results', 'ece_scannet.svg'))
print('=> figure generated.')