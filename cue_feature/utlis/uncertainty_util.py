import numpy as np
import torch
from torch_cluster import knn
import matin
from itertools import cycle
from matplotlib import pyplot as plt
from os.path import join, exists, dirname

plt.style.use('ggplot')

def get_matches(feat_source, feat_target, sym=False):
    matches = knn(feat_target, feat_source, k=1).T
    if sym:
        match_inv = knn(feat_source, feat_target, k=1).T
        mask = match_inv[matches[:, 1], 1] == torch.arange(matches.shape[0], device=feat_source.device)
        return matches[mask]
    else:
        return matches


def compute_hit_ratio(xyz, xyz_target, T_gt, tau_1):
    """
    compute proportion of point which are close.
    """
    assert xyz.shape == xyz.shape
    dist = torch.norm(xyz.mm(T_gt[:3, :3].T) + T_gt[:3, 3] - xyz_target, dim=1)

    return dist < tau_1


def get_bins(sigma, num_of_bins=11):
    sigma = sigma.flatten()
    if not isinstance(sigma, np.ndarray):    
        sigma = sigma.cpu().numpy()
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


def parse_hr_uncertainty(feat, feat_target, xyz, xyz_target, sigma, sigma_target, T_gt, tau1):
    """
    ===input=== (all tensor)
    feat: [N1, 32]
    feat_target: [N2, 32]
    xyz: [N1, 3]
    xyz_target: [N2, 3]
    sigma: [N1, 1]
    sigma_target: [N2, 1]
    tau1: [1,]
    ===return=== (all numpy)
    bin_hit_ratios_pair: [10, ]
    bin_counts_pair: [10, ]
    hr: [1,]
    """
    matches_pred = get_matches(feat, feat_target, sym=True)
    hit_map = compute_hit_ratio(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], T_gt, tau1)
    hit_map = hit_map.cpu().numpy()
    matches_pred = matches_pred.cpu().numpy()
    whole_correct_inds = np.arange(len(hit_map))[hit_map]
    # pair ECE
    pair_sigma = sigma[matches_pred[:, 0]] + sigma_target[matches_pred[:, 1]]
    split_inds, bins = get_bins(pair_sigma)                # in increasing order
    bin_counts_pair = np.array([len(x) for x in split_inds])
    bin_hit_ratios_pair = np.zeros((len(split_inds)))
    for i, bin_inds in enumerate(split_inds):
        if len(bin_inds) == 0:
            bin_hit_ratios_pair[i] = 0
            continue
        bin_correct_inds = np.intersect1d(bin_inds, whole_correct_inds)
        bin_hit_ratios_pair[i] = len(bin_correct_inds) / len(bin_inds)

    return bin_hit_ratios_pair, bin_counts_pair, hit_map.mean()


def get_bin_hr_avg(bin_hit_ratios, bin_counts, st=0, end=10):
    """
    bin_hit_ratios: (1, 10)
    bin_counts: (1, 10)
    """
    avg = bin_hit_ratios[st:end] * bin_counts[st:end] / bin_counts[st:end].sum()
    return avg.sum()


def get_samples_hr_avg(bin_hit_ratios, bin_counts, st=0, end=10):
    """
    bin_hit_ratios: (100, 10)
    bin_counts: (100, 10)
    """
    avg = bin_hit_ratios[:, st:end] * bin_counts[:, st:end] / bin_counts[:, st:end].sum(axis=1, keepdims=True)
    avg = avg.sum(axis=1)
    return avg


# hit_ratio_5 = get_bin_hr_avg(bin_hit_ratios_pairs[0], bin_hit_ratios_pairs_counts[0], 0, 5)
# hit_ratios_5 = get_samples_hr_avg(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, 0, 5)


def output_fmr_threshold_bins(bin_hit_ratios_pairs, hit_ratios_base, hr_dir=None):
    """
    hit_ratios_base: (100, 1)
    bin_hit_ratios_pairs: (100, 10)
    """

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    thre_recall0 = np.zeros((20, 2))
    thre_recall1 = np.zeros((10, 20, 2))
    for j in range(10):
        for i, thre in enumerate(np.linspace(0, 1, 21)[:-1]):
            thre_recall0[i] = np.array([thre, (hit_ratios_base > thre).mean()])
            thre_recall1[j, i, :] = np.array([thre, (bin_hit_ratios_pairs[:, j] > thre).mean()])

    fig, axs = plt.subplots(1, 1, figsize=(5, 5), squeeze=False, dpi=200)
    ax = axs[0][0]
    ax.plot(thre_recall0[:, 0], thre_recall0[:, 1], marker='o', markersize=2, lw=1, label='Random', c='black')
    for j in range(10):
        ax.plot(thre_recall1[j, :, 0], thre_recall1[j, :, 1], marker='o', markersize=2, lw=1, label=f'UL: {j}', linestyle=next(linecycler))
    ax.set_xlabel('Hit Ratio Threshold')
    ax.set_ylabel('Feature Matching Recall')
    matin.ax_default_style(ax, show_grid=True, show_legend=True)
    matin.ax_lims(ax, interval_xticks=0.1)
    if hr_dir is not None:
        plt.savefig(join(hr_dir, 'fmr_threhold_bin.png'), bbox_inches='tight')
    plt.close()
    return fig


def output_fmr_threshold_bins_avg(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hit_ratios_base, ends, hr_dir=None):
    """
    hit_ratios_base: (100, 1)
    bin_hit_ratios_pairs: (100, 10)
    """

    lines = ["-", "--", "-.", ":"]
    linecycler = cycle(lines)

    start = 0
    ends = np.array(ends)
    thre_recall0 = np.zeros((20, 2))
    thre_recall1 = np.zeros((len(ends), 20, 2))
    for j, end in enumerate(ends):
        for i, thre in enumerate(np.linspace(0, 1, 21)[:-1]):
            thre_recall0[i] = np.array([thre, (hit_ratios_base > thre).mean()])
            hit_ratios_ = get_samples_hr_avg(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, start, end)
            thre_recall1[j, i, :] = np.array([thre, (hit_ratios_ > thre).mean()])

    fig, axs = plt.subplots(1, 1, figsize=(5, 5), squeeze=False, dpi=200)
    ax = axs[0][0]
    for j, end in enumerate(ends):
        ax.plot(thre_recall1[j, :, 0], thre_recall1[j, :, 1], marker='o', lw=1, markersize=2, label=f'UL: [{start}:{end}]', linestyle=next(linecycler))
    ax.plot(thre_recall0[:, 0], thre_recall0[:, 1], marker='o', lw=1, markersize=2, label='Random', c='black')
    ax.set_xlabel('Hit Ratio Threshold')
    ax.set_ylabel('Feature Matching Recall')
    matin.ax_default_style(ax, show_grid=True, show_legend=True)
    matin.ax_lims(ax, interval_xticks=0.1)
    if hr_dir is not None:
        plt.savefig(join(hr_dir, 'fmr_threhold_avg.png'), bbox_inches='tight')
    plt.close()

    return fig


def output_hr_bins(bin_hit_ratios_pairs, bin_hit_ratios_pairs_counts, hr, hr_dir=None):
    bin_hit_ratios_pair_avg = bin_hit_ratios_pairs.mean(axis=0, keepdims=True)[0]
    bin_hit_ratios_pairs_counts_avg = bin_hit_ratios_pairs_counts.mean(axis=0, keepdims=True)[0]
    ece = cal_ece(bin_hit_ratios_pair_avg, bin_hit_ratios_pairs_counts_avg)
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), sharex=True, squeeze=False, dpi=200)
    ax = axs[0][0]
    ax.plot(np.arange(len(bin_hit_ratios_pair_avg)), bin_hit_ratios_pair_avg, marker='o')
    ax.plot([0, 9], [hr, hr], linestyle='--', lw=1, alpha=0.5, c='black')
    # ax.plot([0, 9], [1, 0.1], linestyle='--', lw=1, alpha=0.5, c='purple')
    ax.text(0, hr, f'AvgHR={hr:.3f}, ECE:{ece:.3f}')
    ax.set_ylabel('Hit Ratio')
    matin.ax_default_style(ax, ratio=0.7, show_grid=True)
    matin.ax_lims(ax, interval_xticks=1)

    ax = axs[1][0]
    ax.bar(np.arange(len(bin_hit_ratios_pairs_counts_avg)), bin_hit_ratios_pairs_counts_avg)
    ax.set_xlabel('Uncertainty Level')
    ax.set_ylabel('Sample Density')
    matin.ax_default_style(ax, ratio=0.65)
    # plt.subplots_adjust(hspace=-0.45)
    # plt.savefig(join(hr_dir, 'ece.png'), bbox_inches='tight')
    if hr_dir is not None:
        plt.savefig(join(hr_dir, 'ece.png'))
    plt.close()
    return fig


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