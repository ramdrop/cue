#%%
"""
Analysis
"""
from os.path import join, exists, dirname
import os

import numpy as np
np.set_printoptions(precision=3)

import sys
sys.path.append('..')
DIR = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.join(DIR, "..", "..")
sys.path.insert(0, ROOT)

import pickle
from matplotlib import pyplot as plt
plt.style.use('ggplot')

import matin
from utlis import uncertainty_util
# -------------------------------- import end -------------------------------- #


feature_dir = '/LOCAL2/ramdrop/github/point_registration/DPR/logs_modelnet/BTL_0522_095054/eval_val_0523_144457/features'
tau1 = 0.01

eval_dir = dirname(feature_dir)
hr_dir = join(eval_dir, 'hr')
if not exists(hr_dir):
    os.makedirs(hr_dir)
input_file = join(hr_dir, f"hr_uncertainty_{tau1*100:.0f}.pickle")

with open(input_file, 'rb') as handle:
    bin_hit_ratios_pairs = pickle.load(handle)
    bin_hit_ratios_pairs_counts = pickle.load(handle)
    hit_ratios = pickle.load(handle)


fig1 = uncertainty_util.output_hr_bins(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios.mean(), hr_dir)
fig2 = uncertainty_util.output_fmr_threshold_bins(bin_hit_ratios_pairs, hit_ratios, hr_dir)
fig3 = uncertainty_util.output_fmr_threshold_bins_avg(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios, [1, 2, 3, 4, 5, 6], hr_dir)

FMR_at_005 = (hit_ratios > 0.05).mean()
FMR_at_020 = (hit_ratios > 0.20).mean()
print(f'FMR@0.05={FMR_at_005:.3f}, FMR@0.20={FMR_at_020:.3f}')
#%%
debug = True
if debug:
    ece_details_dir = join(hr_dir, 'ece_details')
    if not exists(ece_details_dir):
        os.makedirs(ece_details_dir)
    for data_ind in range(len(hit_ratios)):
        hr_unit = bin_hit_ratios_pairs[data_ind]
        count_unit = bin_hit_ratios_pairs_counts[data_ind]
        fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True, squeeze=False, dpi=200)
        ax = axs[0][0]
        ax.plot(np.arange(len(hr_unit)), hr_unit, marker='o')
        hr_mean = get_bin_hr_avg(hr_unit, count_unit)
        ax.plot([0, 9], [hr_mean, hr_mean], linestyle='--', alpha=0.5, c='black')
        ax.text(5, hr_mean + 0.01, f'Avg Hit Ratio={hr_mean:.3f}')
        ax.set_ylabel('Hit Ratio')
        # ax.set_title(f'Avg Hit Ratio = {hit_ratios.mean():.3f}', fontsize=12)
        matin.ax_default_style(ax, ratio=0.7, show_grid=True)
        matin.ax_lims(ax, interval_xticks=1)
        ax = axs[1][0]
        ax.bar(np.arange(len(count_unit)), count_unit)
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Sample Density')
        matin.ax_default_style(ax, ratio=0.65)
        plt.subplots_adjust(hspace=-0.45)
        plt.savefig(join(ece_details_dir, f'ece_{data_ind}.png'), bbox_inches='tight')
        plt.close()
