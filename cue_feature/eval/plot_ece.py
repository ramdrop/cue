#%%
import sys; sys.path.append('..')
import copy
from tqdm import tqdm
# import pickle
import pickle5 as pickle
import numpy as np
from os.path import dirname, join
from matplotlib import pyplot as plt
from matplotlib import patches
import matin
np.set_printoptions(precision=4, suppress=True)

mc_pickle_file = './logs/HCL_MC_0420_143013/eval_02/ece_results.pickle'
cue_pickle_file = './logs/BTL_0804_095744/eval_0808_042602/ece_results.pickle'
cueplus_pickle_file = './logs/MBTL_0805_024054/eval_0808_073320/ece_results.pickle'

def cal_ece(plist, nlist):
    """
    ===input===
    plist: [10, 1]
    nlist: [10, 1]
    ===return===
    ece: [1]
    """
    clist = np.arange(1, 0, -0.1)
    nlist = nlist / nlist.sum()
    ece_list = nlist * np.abs(clist - plist)
    ece = ece_list.sum()
    return ece


#%%

# MCD
with open(mc_pickle_file, 'rb') as handle:
    fmr_sets = pickle.load(handle)
    query_bin_hr = pickle.load(handle)
    query_bin_counts = pickle.load(handle)
    pair_bin_hr = pickle.load(handle)
    pair_bin_counts = pickle.load(handle)
    hr = pickle.load(handle)
    sigma_mcd = pickle.load(handle)
    correct_inds_mcd = pickle.load(handle)

hr = np.concatenate([x[1] for x in hr])
pair_bin_hr_mcd = np.concatenate([x[1] for x in pair_bin_hr], axis=0)
pair_bin_counts_mcd = np.concatenate([x[1] for x in pair_bin_counts], axis=0)
ece_mcd = cal_ece(pair_bin_hr_mcd.mean(axis=0), pair_bin_counts_mcd.mean(axis=0))
print(f'ece_mcd:{ece_mcd:.3f}')
FMR_at_005 = (hr>0.05).mean()
FMR_at_020 = (hr>0.20).mean()
print(f'FMR@0.05={FMR_at_005:.3f}\nFMR@0.20={FMR_at_020:.3f}')


# CUE
with open(cue_pickle_file, 'rb') as handle:
    fmr_sets = pickle.load(handle)
    query_bin_hr = pickle.load(handle)
    query_bin_counts = pickle.load(handle)
    pair_bin_hr = pickle.load(handle)
    pair_bin_counts = pickle.load(handle)
    hr = pickle.load(handle)
    sigma_cue = pickle.load(handle)
    correct_inds_cue = pickle.load(handle)

hr = np.concatenate([x[1] for x in hr])
pair_bin_hr_cue = np.concatenate([x[1] for x in pair_bin_hr], axis=0)
pair_bin_counts_cue = np.concatenate([x[1] for x in pair_bin_counts], axis=0)
ece_cue = cal_ece(pair_bin_hr_cue.mean(axis=0), pair_bin_counts_cue.mean(axis=0))
print(f'ece_cue:{ece_cue:.3f}')
FMR_at_005 = (hr>0.05).mean()
FMR_at_020 = (hr>0.20).mean()
print(f'FMR@0.05={FMR_at_005:.3f}\nFMR@0.20={FMR_at_020:.3f}')


# CUE+
with open(cueplus_pickle_file, 'rb') as handle:
    fmr_sets = pickle.load(handle)
    query_bin_hr = pickle.load(handle)
    query_bin_counts = pickle.load(handle)
    pair_bin_hr = pickle.load(handle)
    pair_bin_counts = pickle.load(handle)
    hr = pickle.load(handle)
    sigma_lrmg = pickle.load(handle)
    correct_inds_lrmg = pickle.load(handle)

hr = np.concatenate([x[1] for x in hr])
pair_bin_hr_lrmg = np.concatenate([x[1] for x in pair_bin_hr], axis=0)
pair_bin_counts_lrmg = np.concatenate([x[1] for x in pair_bin_counts], axis=0)
ece_lrmg = cal_ece(pair_bin_hr_lrmg.mean(axis=0), pair_bin_counts_lrmg.mean(axis=0))
print(f'ece_LRMG:{ece_lrmg:.3f}')
FMR_at_005 = (hr > 0.05).mean()
FMR_at_020 = (hr > 0.20).mean()
print(f'FMR@0.05={FMR_at_005:.3f}\nFMR@0.20={FMR_at_020:.3f}')


# COPY FROM ECE_BASELINE ============= #
baseline_hr = np.array([0.553, 0.553, 0.556, 0.552, 0.556, 0.556, 0.554, 0.556, 0.552, 0.554])
baseline_counts = np.array([75.872, 74.596, 74.911, 74.937, 74.69, 74.928, 74.791, 74.711, 74.58, 75.866])

labels = ['FCGF+RG', 'FCGF+MCD', 'FCGF+CUE', 'FCGF+CUE+']
colors = ['#82B0D2', '#73A9AD', '#FFC4C4', '#B25068']
markersize = 4

fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True, squeeze=False, dpi=200)
ax = axs[0][0]
ax.plot(np.arange(len(pair_bin_hr_mcd.mean(axis=0))) / 10, 1 - np.arange(len(pair_bin_hr_mcd.mean(axis=0))) / 10, marker='o', markersize=0, linewidth=1, color='black', linestyle='--', alpha=0.7, label='ideal')
ax.plot(np.arange(len(pair_bin_hr_lrmg.mean(axis=0))) / 10, baseline_hr, marker='o', markersize=markersize, label=labels[0],linewidth=2, color=colors[0])
ax.plot(np.arange(len(pair_bin_hr_mcd.mean(axis=0))) / 10, pair_bin_hr_mcd.mean(axis=0), marker='o', markersize=markersize,linewidth=2, label=labels[1], color=colors[1])
ax.plot(np.arange(len(pair_bin_hr_cue.mean(axis=0))) / 10, pair_bin_hr_cue.mean(axis=0), marker='o', markersize=markersize, linewidth=2,label=labels[2], color=colors[2])
ax.plot(np.arange(len(pair_bin_hr_lrmg.mean(axis=0))) / 10, pair_bin_hr_lrmg.mean(axis=0), marker='o', markersize=markersize,linewidth=2, label=labels[3], color=colors[3])
ax.set_ylabel('Hit Ratio')
matin.ax_default_style(ax, ratio=0.75, show_grid=True, show_legend=True)
matin.ax_lims(ax, interval_xticks=1)

ax = axs[1][0]
width = 0.02
delta = 0.015
ax.bar(np.arange(len(pair_bin_counts_mcd.mean(axis=0))) / 10 , baseline_counts / baseline_counts.sum(), width=width, color=colors[0])
ax.bar(np.arange(len(pair_bin_counts_mcd.mean(axis=0))) / 10 + delta, pair_bin_counts_mcd.mean(axis=0) / pair_bin_counts_mcd.mean(axis=0).sum(), width=width, color=colors[1])
ax.bar(np.arange(len(pair_bin_counts_cue.mean(axis=0))) / 10 + 2*delta, pair_bin_counts_cue.mean(axis=0) / pair_bin_counts_cue.mean(axis=0).sum(), width=width, color=colors[2])
ax.bar(np.arange(len(pair_bin_counts_lrmg.mean(axis=0))) / 10 + 3 * delta, pair_bin_counts_lrmg.mean(axis=0) / pair_bin_counts_lrmg.mean(axis=0).sum(), width=width, color=colors[3])
ax.set_xlabel('Uncertainty Level')
ax.set_ylabel('Sample Density')
matin.ax_default_style(ax, ratio=0.37)
plt.subplots_adjust(hspace=-0.6)
ax.set_xticks(np.arange(0, 1, 0.1))
# ax.set_xticklabels(['1.0', '0.9', '0.8', '0.7', '0.6', '0.5', '0.4', '0.3', '0.2', '0.1'])
ax.set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0'])
plt.savefig(join('logs/results', 'ece_fcgf.png'), bbox_inches='tight')
plt.savefig(join('logs/results', 'ece_fcgf.svg'), bbox_inches='tight')
