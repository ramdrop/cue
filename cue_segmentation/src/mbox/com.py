import numpy as np
from scipy.special import softmax
from matplotlib import pyplot as plt
from os.path import join, dirname, exists
import matin

def get_eval_dic():
    eval_dir_dic = {
        'cue': 'logs_scannet/2minkprob_0623_213340/eval_0712_132839',
        'lrmg': 'logs_scannet/minkprobmg_0805_084511/eval_0815_093715',
        'entropy': 'logs_scannet/2mink_0623_203059/eval_0828_112758',
        # 'entropy21': 'logs_scannet/minkprob_0714_023141/eval_0715_065601',
        # 'entropy21cue': 'logs_scannet/minkprob_0705_065734/eval_0712_124534',
        # 'aleatoric': 'logs_scannet/minkaleatoric_0729_142245/eval_0801_031100',                # softplus, 10
        'aleatoric': 'logs_scannet/minkaleatoric_0803_080321/eval_0808_030150',                  # sigmoid, 40
        'mc': 'logs_scannet/minkmc_0801_123159/eval_0803_032138',                                # p=0.2
        'mc_01': 'logs_scannet/minkmc_0817_065249/eval_0826_034933',                             # p=0.1
        'mc_005': 'logs_scannet/minkmc_0817_065324/eval_0826_035015',                            # p=0.05
        'dul': 'logs_scannet/minkdul_1230_095824/eval_0104_112122',
        'rul': 'logs_scannet/minkrul_0107_102653/eval_0109_102533',
    }
    return eval_dir_dic



def load_meta(label_file, *args):
    data_pack = []
    for item in args:
        try:
            data = np.load(label_file.replace('label', item))
        except:
            data = -1
        data_pack.append(data)
    return data_pack


def entropy(p, dim=-1, keepdims=True):
    # exactly the same as scipy.stats.entropy()
    return -np.where(p > 0, p * np.log(p), [0.0]).sum(axis=dim, keepdims=keepdims)

def score_to_entropy(seg):
    if type(seg) is not np.ndarray:
        seg = seg.cpu().numpy()
    seg_prob = softmax(seg, axis=1)
    return entropy(seg_prob) / np.log(seg.shape[-1])


def get_bins_eqsample(sigma, num_of_bins=10):
    if sigma.shape[-1] == 1:
        sigma = sigma.flatten()
    sorted_ind = np.argsort(sigma)
    indices = np.array_split(sorted_ind, num_of_bins)
    return indices, None

def get_bins(sigma, num_of_bins=11):
    if sigma.shape[-1] == 1:
        sigma = sigma.flatten()
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


def get_bins_precision(sigma, pred, label, prune_ignore=False):
    if type(sigma) is not np.ndarray:
        sigma = sigma.cpu().numpy()
    if type(pred) is not np.ndarray:
        pred = pred.cpu().numpy()
    if type(label) is not np.ndarray:
        label = label.cpu().numpy()
    split_inds, bins = get_bins(sigma)
    bin_counts = np.array([len(x) for x in split_inds])
    bin_precisions = np.zeros((len(split_inds)))
    for i, bin_inds in enumerate(split_inds):
        if len(bin_inds) == 0:
            bin_precisions[i] = 0
            continue
        pred_ = pred[bin_inds]
        label_ = label[bin_inds]
        correct = pred_ == label_
        if prune_ignore:
            correct = correct[label_ != 255]
        if correct.size > 0:
            precision_ = correct.sum() / correct.size
        else:
            precision_ = 0
        bin_precisions[i] = precision_

    correct = pred == label
    if prune_ignore:
        correct = correct[label != 255]
    if correct.size > 0:
        precision = correct.sum() / correct.size
    else:
        precision = 0
    return bin_precisions, bin_counts, precision


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


def vis_uncertainty_precision(bin_precisions, bin_cnts, avg_precision, filename=None):

    ece = cal_ece(bin_precisions, bin_cnts)
    fig, axs = plt.subplots(2, 1, figsize=(5, 6), sharex=True, squeeze=False, dpi=200)
    ax = axs[0][0]
    ax.plot(np.arange(len(bin_precisions)), bin_precisions, marker='o')
    ax.plot([0, 9], [avg_precision, avg_precision], linestyle='--', lw=1, alpha=0.5, c='black')
    # ax.plot([0, 9], [1, 0.1], linestyle='--', lw=1, alpha=0.5, c='purple')
    ax.text(0, avg_precision, f'AvgPrecision={avg_precision:.3f}, ECE:{ece:.3f}')
    ax.set_ylabel('Precision')
    matin.ax_default_style(ax, ratio=0.7, show_grid=True)
    matin.ax_lims(ax, interval_xticks=1)

    ax = axs[1][0]
    ax.bar(np.arange(len(bin_cnts)), bin_cnts)
    ax.set_xlabel('Uncertainty Level')
    ax.set_ylabel('Sample Density')
    matin.ax_default_style(ax, ratio=0.65)
    # plt.subplots_adjust(hspace=-0.45)
    # plt.savefig(join(hr_dir, 'ece.png'), bbox_inches='tight')
    if filename is not None:
        plt.savefig(filename)
    plt.close()
    return fig