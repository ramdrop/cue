#%%
"""
Calculate Random
"""

import sys; sys.path.append('..')
import copy
from tqdm import tqdm
import pickle5 as pickle
import numpy as np
from os.path import dirname, join
from matplotlib import pyplot as plt
from matplotlib import patches
import matin

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


def get_bins(sigma, num_of_bins=11):
    sigma_min = np.min(sigma)
    sigma_max = np.max(sigma)
    # print(sigma_min, sigma_max)
    bins = np.linspace(sigma_min, sigma_max, num=num_of_bins)
    indices = []
    for index in range(num_of_bins - 1):
        target_q_ind_l = np.where(sigma >= bins[index])
        if index != num_of_bins - 2:
            target_q_ind_r = np.where(sigma < bins[index + 1])
        else:
            # the last bin use close interval
            target_q_ind_r = np.where(sigma <= bins[index + 1])
        target_q_ind = np.intersect1d(target_q_ind_l, target_q_ind_r)
        indices.append(target_q_ind)
    # print([len(x) for x in indices])
    return indices, bins


baseline_pickle_file = './logs/HCL_0419_161400/eval_0811_085613/ece_results.pickle'
with open(baseline_pickle_file, 'rb') as handle:
    fmr_sets = pickle.load(handle)
    query_bin_hr = pickle.load(handle)
    query_bin_counts = pickle.load(handle)
    pair_bin_hr = pickle.load(handle)
    pair_bin_counts = pickle.load(handle)
    hr = pickle.load(handle)
    sigma_baseline = pickle.load(handle)
    correct_inds = pickle.load(handle)

hr = np.concatenate([x[1] for x in hr])
FMR_at_005 = (hr>0.05).mean()
FMR_at_020 = (hr>0.20).mean()
print(f'FMR@0.05={FMR_at_005:.3f}\nFMR@0.20={FMR_at_020:.3f}')

# correct_inds[0][0]
# len(correct_inds[0][1])                # 506
# correct_inds[0][1][0].shape            # (1559,)


sigma = sigma_baseline


# CALCULATE HR ======================= #
percentages = [1, 0.75, 0.5, 0.25]
random_all = []
for i in tqdm(range(len(sigma))):                                                                  # scenes: 1,2,...,8
    sigma_scene = sigma[i][1]
    correct_inds_scene = correct_inds[i][1]
    for j in range(len(sigma_scene)):                                                              # frames: 1,2,..,500
        sigma_single = sigma_scene[j]
        correct_inds_single = correct_inds_scene[j]
        random_box = []
        for percent in percentages:
            N = len(sigma_single)
            all_inds = np.arange(N)
            random_inds = np.random.choice(all_inds, int(percent * N), replace=False)              # sigma_copy = copy.deepcopy(sigma_single)
            precision_random = len(np.intersect1d(random_inds, correct_inds_single)) / N           # print(f"percent={percent:.3f}, precision_random={precision_random:.3f},precision_cue={precision_cue:.3f}")
            random_box.append(precision_random)
        random_all.append(random_box)
random_all = np.array(random_all)
print(f"@%:{percentages}")
print(f"HR={random_all.mean(axis=0)}")


# CALCULATE ECE ====================== #
pair_bin_hr = []
pair_bin_counts = []
for i in tqdm(range(len(sigma))):      # scenes: 1,2,...,8
    sigma_scene = sigma[i][1]
    correct_inds_scene = correct_inds[i][1]
    for j in range(len(sigma_scene)):  # frames: 1,2,..,500
        sigma_single = sigma_scene[j]
        correct_inds_single = correct_inds_scene[j]
        N = len(sigma_single)
        all_inds = np.arange(N)
        sigma_single = np.random.rand(N)

        # pair ECE
        split_inds, bins = get_bins(sigma_single)          # in increase order
        pair_bin_hr_ = np.zeros((len(split_inds)))
        pair_bin_counts_ = np.array([len(x) for x in split_inds])
        for i, bin_inds in enumerate(split_inds):
            if len(bin_inds) == 0:
                pair_bin_hr[i] = 0
                continue
            bin_correct_inds = np.intersect1d(bin_inds, correct_inds_single)
            pair_bin_hr_[i] = len(bin_correct_inds) / len(bin_inds)
        pair_bin_hr.append(pair_bin_hr_)
        pair_bin_counts.append(pair_bin_counts_)
pair_bin_hr = np.array(pair_bin_hr)
pair_bin_counts = np.array(pair_bin_counts)
print(f"HR={pair_bin_hr.mean(axis=0)}")
print(f"Counts={pair_bin_counts.mean(axis=0)}")
ece_random = cal_ece(pair_bin_hr.mean(axis=0), pair_bin_counts.mean(axis=0))
print(f"ECE={ece_random:.3f}")