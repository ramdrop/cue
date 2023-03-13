#%%
import torch
from os.path import join, exists, dirname
import os
import numpy as np
from tqdm import tqdm
from torch_geometric.nn import knn
import open3d
import copy
from glob import glob
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import pickle
import matin
import sys
sys.path.append('..')
from utlis import pointcloud

plt.style.use('ggplot')
import importlib

importlib.reload(matin)


def get_matrix_system(xyz, xyz_target, weight):
    """
    Build matrix of size 3N x 6 and b of size 3N

    xyz size N x 3
    xyz_target size N x 3
    weight size N
    the matrix is minus cross product matrix concatenate with the identity (rearanged).
    """
    assert xyz.shape == xyz_target.shape
    A_x = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    A_y = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    A_z = torch.zeros(xyz.shape[0], 6, device=xyz.device)
    b_x = weight.view(-1) * (xyz_target[:, 0] - xyz[:, 0])
    b_y = weight.view(-1) * (xyz_target[:, 1] - xyz[:, 1])
    b_z = weight.view(-1) * (xyz_target[:, 2] - xyz[:, 2])
    A_x[:, 1] = weight.view(-1) * xyz[:, 2]
    A_x[:, 2] = -weight.view(-1) * xyz[:, 1]
    A_x[:, 3] = weight.view(-1) * 1
    A_y[:, 0] = -weight.view(-1) * xyz[:, 2]
    A_y[:, 2] = weight.view(-1) * xyz[:, 0]
    A_y[:, 4] = weight.view(-1) * 1
    A_z[:, 0] = weight.view(-1) * xyz[:, 1]
    A_z[:, 1] = -weight.view(-1) * xyz[:, 0]
    A_z[:, 5] = weight.view(-1) * 1
    return torch.cat([A_x, A_y, A_z], 0), torch.cat([b_x, b_y, b_z], 0).view(-1, 1)


def get_cross_product_matrix(k):
    return torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=k.device)


def rodrigues(axis, theta):
    """
    given an axis of norm one and an angle, compute the rotation matrix using rodrigues formula
    source : https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    K = get_cross_product_matrix(axis)
    t = torch.tensor([theta], device=axis.device)
    R = torch.eye(3, device=axis.device) + torch.sin(t) * K + (1 - torch.cos(t)) * K.mm(K)
    return R


def get_trans(x):
    """
    get the rotation matrix from the vector representation using the rodrigues formula
    """
    T = torch.eye(4, device=x.device)
    T[:3, 3] = x[3:]
    axis = x[:3]
    theta = torch.norm(axis)
    if theta > 0:
        axis = axis / theta
    T[:3, :3] = rodrigues(axis, theta)
    return T


def get_geman_mclure_weight(xyz, xyz_target, mu):
    """
    compute the weights defined here for the iterative reweighted least square.
    http://vladlen.info/papers/fast-global-registration.pdf
    """
    norm2 = torch.norm(xyz_target - xyz, dim=1)**2
    return (mu / (mu + norm2)).view(-1, 1)


def fast_global_registration(xyz, xyz_target, weight=None, mu_init=1, num_iter=20):
    """
    estimate the rotation and translation using Fast Global Registration algorithm (M estimator for robust estimation)
    http://vladlen.info/papers/fast-global-registration.pdf
    """
    assert xyz.shape == xyz_target.shape

    T_res = torch.eye(4, device=xyz.device)
    mu = mu_init
    source = xyz.clone()

    if weight is None:
        weight = torch.ones(len(source), 1, device=xyz.device)

    # print(weight)
    for i in range(num_iter):
        if i > 0 and i % 5 == 0:
            mu /= 2.0
        A, b = get_matrix_system(source, xyz_target, weight)
        x = torch.linalg.solve(A.T.mm(A), A.T @ b)
        T = get_trans(x.view(-1))
        source = source.mm(T[:3, :3].T) + T[:3, 3]
        T_res = T @ T_res
        weight = get_geman_mclure_weight(source, xyz_target, mu)
    return T_res


def compute_transfo_error(T_gt, T_pred):
    """
    compute the translation error (the unit depends on the unit of the point cloud)
    and compute the rotation error in degree using the formula (norm of antisymetr):
    http://jlyang.org/tpami16_go-icp_preprint.pdf
    """
    rte = torch.norm(T_gt[:3, 3] - T_pred[:3, 3])
    cos_theta = (torch.trace(T_gt[:3, :3].mm(T_pred[:3, :3].T)) - 1) * 0.5
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    rre = torch.acos(cos_theta) * 180 / np.pi
    return rte, rre


def estimate_transfo(xyz, xyz_target):
    """
    estimate the rotation and translation using Kabsch algorithm
    Parameters:
    xyz :
    xyz_target:
    """
    assert xyz.shape == xyz.shape
    xyz_c = xyz - xyz.mean(0)
    xyz_target_c = xyz_target - xyz_target.mean(0)
    Q = xyz_c.T.mm(xyz_target_c) / len(xyz)
    U, S, V = torch.svd(Q)
    d = torch.det(V.mm(U.T))
    diag = torch.diag(torch.tensor([1, 1, d], device=xyz.device))
    R = V.mm(diag).mm(U.T)
    t = xyz_target.mean(0) - R @ xyz.mean(0)
    T = torch.eye(4, device=xyz.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


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


def get_bin_hr_avg(bin_hit_ratios, bin_counts, st=0, end=10):
    """
    bin_hit_ratios: (1, 10)
    bin_counts: (1, 10)
    """
    avg = bin_hit_ratios[st:end] * bin_counts[st:end] / bin_counts[st:end].sum()
    return avg.sum()


def teaser_pp_registration(
    xyz,
    xyz_target,
    noise_bound=0.05,
    cbar2=1,
    rotation_gnc_factor=1.4,
    rotation_max_iterations=100,
    rotation_cost_threshold=1e-12,
):
    assert xyz.shape == xyz_target.shape
    import teaserpp_python

    # Populating the parameters
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = cbar2
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS)
    solver_params.rotation_gnc_factor = rotation_gnc_factor
    solver_params.rotation_max_iterations = rotation_max_iterations
    solver_params.rotation_cost_threshold = rotation_cost_threshold

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)

    solver.solve(xyz.T.detach().cpu().numpy(), xyz_target.T.detach().cpu().numpy())

    solution = solver.getSolution()
    T_res = torch.eye(4, device=xyz.device)
    T_res[:3, :3] = torch.from_numpy(solution.rotation).to(xyz.device)
    T_res[:3, 3] = torch.from_numpy(solution.translation).to(xyz.device)
    return T_res


def ransac_registration(xyz, xyz_target, distance_threshold=0.05):
    """
    use Open3D version of RANSAC
    """
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz.detach().cpu().numpy())

    pcd_t = open3d.geometry.PointCloud()
    pcd_t.points = open3d.utility.Vector3dVector(xyz_target.detach().cpu().numpy())
    rang = np.arange(len(xyz))
    corres = np.stack((rang, rang), axis=1)
    corres = open3d.utility.Vector2iVector(corres)
    result = open3d.pipelines.registration.registration_ransac_based_on_correspondence(
        pcd,
        pcd_t,
        corres,
        distance_threshold,
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),
    )

    return torch.tensor(result.transformation).float()

def ransac_feature(xyz0, xyz1, feat0, feat1, distance_threshold=0.3):
    pcd0 = pointcloud.make_open3d_point_cloud(xyz0.cpu().numpy())
    pcd1 = pointcloud.make_open3d_point_cloud(xyz1.cpu().numpy())
    feat0 = pointcloud.make_open3d_feature(feat0, feat0.shape[1], feat0.shape[0])
    feat1 = pointcloud.make_open3d_feature(feat1, feat1.shape[1], feat1.shape[0])

    ransac_result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=pcd0,
        target=pcd1,
        source_feature=feat0,
        target_feature=feat1,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),                  # 4000000, 500
    )

    T_ransac = torch.from_numpy(ransac_result.transformation.astype(np.float32))
    return T_ransac

def weighted_procrustes(X, Y, w, eps=1e-7):
    '''
    X: torch tensor N x 3
    Y: torch tensor N x 3
    w: torch tensor N
    '''

    # https://ieeexplore.ieee.org/document/88573
    assert len(X) == len(Y)
    if w is None:
        w = torch.ones(len(X), 1, device=xyz.device)

    W1 = torch.abs(w).sum()
    w_norm = w / (W1 + eps)
    mux = (w_norm * X).sum(0, keepdim=True)
    muy = (w_norm * Y).sum(0, keepdim=True)

    # Use CPU for small arrays
    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()                  # (3, 3) = (3, N) x (N, 3)
    safety_svd = eps * torch.eye(3).to(Sxy)
    # U, D, V = Sxy.svd()
    U, D, V = torch.svd(Sxy + safety_svd)
    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t())).float()
    t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
    T = torch.eye(4, device=xyz.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def align(xyz, xyz_target, matches_pred, feat, feat_target, method='FGR', weight=None):
    if weight is not None:
        weight = (weight - weight.min()) / (weight.max() - weight.min())

    if method == 'FGR':
        T_est = fast_global_registration(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], weight)
    elif method == 'TEASER':
        T_est = teaser_pp_registration(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], noise_bound=0.1)
    elif method == 'RANSAC':
        T_est = ransac_registration(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]])
    elif method == 'RANSAC_FEAT':
        # T_est = ransac_feature(xyz, xyz_target, feat, feat_target)
        T_est = ransac_feature(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], feat[matches_pred[:, 0]], feat_target[matches_pred[:, 1]])
    elif method == 'PRO':
        T_est = weighted_procrustes(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], weight)

    return T_est


#%%
"""
Registeration with sampled points
"""
tau1 = 0.01
tau2 = 0.05
st, end = 0, 5
method = 'FGR'                         # FGR | TEASER | RANSAC | PRO | RANSAC_FEAT
use_random_points = True
cut_matches = True
use_cuda = not use_random_points
# ------------------------------------- - ------------------------------------ #

np.set_printoptions(precision=3, suppress=True)
np.random.seed(0)
torch.manual_seed(0)

feature_dir = '/LOCAL2/ramdrop/github/point_registration/DPR/logs_modelnet/BTL_0519_093333/eval_test_0519_150103/features'
eval_dir = dirname(feature_dir)
reg_dir = join(eval_dir, 'registration')
if not exists(reg_dir):
    os.makedirs(reg_dir)

feature_files = glob(join(feature_dir, '*.npz'))
feature_files.sort(key=lambda x: int(x[-9:-4]))

base_error = []
bins_error = []
rows = []

# data_inds = range(len(feature_files))
data_inds = np.random.choice(len(feature_files), 100, replace=False)
# data_inds = data_inds[:5]

exp_name = f"RTE_RRE_{method}_{st}_{end}_{'a' if use_cuda else ''}{'c' if cut_matches else ''}_{len(data_inds)}"
output_file = join(reg_dir, f"{exp_name}.csv")
print(exp_name)
for data_ind in tqdm(data_inds):
    try:
        data_pack = np.load(join(feature_dir, f'{data_ind:0>5d}' + '.npz'))
    except:
        print(f'skip {data_ind}.npz')
        continue
    xyz, xyz_target, T_gt, feat, feat_target, sigma, sigma_target = \
                data_pack['xyz'], data_pack['xyz_target'], data_pack['T_gt'],\
                data_pack['feat'], data_pack['feat_target'], data_pack['sigma'], data_pack['sigma_target']

    device = torch.device('cuda' if use_cuda else 'cpu')                                           # cpu | cuda
    xyz = torch.from_numpy(xyz).to(device)
    xyz_target = torch.from_numpy(xyz_target).to(device)
    T_gt = torch.from_numpy(T_gt).to(device)
    feat = torch.from_numpy(feat).to(device)
    feat_target = torch.from_numpy(feat_target).to(device)
    num_pt, num_pt_target = xyz.shape[0], xyz_target.shape[0]

    # 1. use 5000 random points
    if use_random_points:
        rand = torch.randperm(len(feat))[:10000]
        rand_target = torch.randperm(len(feat_target))[:10000]
        matches_pred = get_matches(feat[rand], feat_target[rand_target], sym=True)
        xyz, xyz_target = xyz[rand], xyz_target[rand_target]
        sigma, sigma_target = sigma[rand], sigma_target[rand_target]
    else:
        # 2. use all points
        matches_pred = get_matches(feat, feat_target, sym=True)

    hit_map = compute_hit_ratio(xyz[matches_pred[:, 0]], xyz_target[matches_pred[:, 1]], T_gt, tau1)
    hit_map = hit_map.cpu().numpy()
    matches_pred = matches_pred.cpu().numpy()

    whole_correct_inds = np.arange(len(hit_map))[hit_map]
    hr = hit_map.mean()
    pair_sigma = sigma[matches_pred[:, 0]] + sigma_target[matches_pred[:, 1]]
    # pair_sigma = sigma[rand[matches_pred[:, 0]]] + sigma_target[rand_target[matches_pred[:, 1]]]
    split_inds, bins = get_bins(pair_sigma.flatten())      # in an increasing order
    bin_counts_pair = np.array([len(x) for x in split_inds])
    bin_hit_ratios_pair = np.zeros((len(split_inds)))
    for i, bin_inds in enumerate(split_inds):
        if len(bin_inds) == 0:
            bin_hit_ratios_pair[i] = 0
            continue
        bin_correct_inds = np.intersect1d(bin_inds, whole_correct_inds)
        bin_hit_ratios_pair[i] = len(bin_correct_inds) / len(bin_inds)

    # --------------------------------- debug -------------------------------- #
    debug = False
    if debug:
        plt.style.use('ggplot')
        fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True, squeeze=False, dpi=200)
        ax = axs[0][0]
        ax.plot(np.arange(len(bin_hit_ratios_pair)), bin_hit_ratios_pair, marker='o')
        ax.set_ylabel('Hit Ratio')
        # ax.set_title(f'Avg Hit Ratio = {hr:.3f}')
        ax.plot([0, 9], [hr, hr], linestyle='--', alpha=0.5, c='black')
        ax.text(5, hr + 0.01, f'Avg Hit Ratio={hr:.3f}')
        matin.ax_default_style(ax, ratio=0.7, show_grid=True)

        ax = axs[1][0]
        ax.bar(np.arange(len(bin_counts_pair)), bin_counts_pair)
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Sample Density')
        matin.ax_default_style(ax, ratio=0.65)
        plt.subplots_adjust(hspace=-0.4)

        pc_details = join(reg_dir, exp_name)
        if not exists(pc_details):
            os.makedirs(pc_details, mode=0o755)
        plt.savefig(join(pc_details, f'ece_{data_ind}.png'), bbox_inches='tight')
        plt.close()

        # map from sampled indices to global indices
        inds = rand[matches_pred[:, 0]]
        inds_target = rand_target[matches_pred[:, 1]]
        matches_pred_bin = matches_pred[np.concatenate(split_inds[st:end])]
        bins_inds = rand[matches_pred_bin[:, 0]]
        bins_inds_target = rand_target[matches_pred_bin[:, 1]]
        with open(join(pc_details, f'{data_ind}.pickle'), 'wb') as handle:
            pickle.dump(inds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(inds_target, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(bins_inds, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(bins_inds_target, handle, protocol=pickle.HIGHEST_PROTOCOL)

        continue
    # ----------------------------------- - ---------------------------------- #

    rte_fgr_gtcor, rre_fgr_gtcor = 0, 0
    # if not use_random_points:
    #     # GT error
    #     T_fgr_gtcor = align(xyz, xyz_target, matches_gt, method).to(device)
    #     rte_fgr_gtcor, rre_fgr_gtcor = compute_transfo_error(T_fgr_gtcor, T_gt)
    #     rte_fgr_gtcor, rre_fgr_gtcor = rte_fgr_gtcor.item(), rre_fgr_gtcor.item()

    # resample stragety 1
    matches_pred_bin = matches_pred[np.concatenate(split_inds[st:end])]

    # resample stragety 2
    matches_cat = np.hstack((matches_pred, pair_sigma))
    if matches_cat.shape[0] % 2:
        matches_cat = matches_cat[:-1]
    matches_cat_group = matches_cat.reshape(-1, 2 ,3)
    sub_ind = np.argsort(matches_cat_group[:, :, -1])
    matches_cat_group_sorted0 = np.array([matches_cat_group[x][sub_ind[x]][0,:] for x in range(matches_cat_group.shape[0])])
    matches_cat_group_sorted1 = np.array([matches_cat_group[x][sub_ind[x]][1,:] for x in range(matches_cat_group.shape[0])])
    matches_pred_bin = matches_cat_group_sorted0

    T_fgr_precor_bin = align(xyz, xyz_target, matches_pred_bin, feat, feat_target, method=method)
    rte_fgr_precor_bin, rre_fgr_precor_bin = compute_transfo_error(T_fgr_precor_bin.to(device), T_gt)
    hr_bins_avg = get_bin_hr_avg(bin_hit_ratios_pair, bin_counts_pair, st=st, end=end)

    # baseline stragety
    if cut_matches:
        inds_base = np.random.choice(matches_pred.shape[0], matches_pred_bin.shape[0], replace=False)
        matches_pred = matches_pred[inds_base]
    T_fgr_precor = align(xyz, xyz_target, matches_pred, feat, feat_target, method=method).to(device)
    rte_fgr_precor, rre_fgr_precor = compute_transfo_error(T_fgr_precor.to(device), T_gt)

    success = 0
    if rte_fgr_precor < 2 and rre_fgr_precor < 5:
        success = 1

    cols = ['data_ind', 'num_pt', 'num_pt_target', 'pred_pair', 'hr', 'bins_pair',f'hr_avg_bins:[{st},{end}]', f'hr_counts_bins:[{st},{end}]', \
        'gt_rte', 'gt_rre', 'baseline_rte', 'baseline_rre', 'bins_rte', 'bins_rre', 'success']
    row = [data_ind, num_pt, num_pt_target, matches_pred.shape[0], f'{hr:.3f}', matches_pred_bin.shape[0], f'{hr_bins_avg:.3f}', bin_counts_pair[st:end].sum(),\
        rte_fgr_gtcor, rre_fgr_gtcor, f'{rte_fgr_precor.item():.3f}', f'{rre_fgr_precor.item():.3f}', f'{rte_fgr_precor_bin.item():.3f}', f'{rre_fgr_precor_bin.item():.3f}', success]
    rows.append(row)

if len(rows) > 0:
    df = pd.DataFrame(data=rows, columns=cols)
    df.to_csv(output_file)
    print(f'=========')
    df = pd.read_csv(output_file)
    success_baseline = np.logical_and(df['baseline_rte'].to_numpy().astype(np.float32) < 2, df['baseline_rre'].to_numpy().astype(np.float32) < 5)
    success_bins = np.logical_and(df['bins_rte'].to_numpy().astype(np.float32) < 2, df['bins_rre'].to_numpy().astype(np.float32) < 5)
    print(
        f"Base: RTE:{df[success_baseline]['baseline_rte'].mean():.3f}+-{df[success_baseline]['baseline_rte'].std():.3f}, RRE:{df[success_baseline]['baseline_rre'].mean():.3f}+-{df[success_baseline]['baseline_rre'].std():.3f}, SR:{success_baseline.sum()}/{df['baseline_rte'].to_numpy().shape[0]}"
    )
    print(
        f"Bins: RTE:{df[success_baseline]['bins_rte'].mean():.3f}+-{df[success_baseline]['bins_rte'].std():.3f}, RRE:{df[success_baseline]['bins_rre'].mean():.3f}+-{df[success_baseline]['bins_rre'].std():.3f}, SR:{success_baseline.sum()}/{df['baseline_rte'].to_numpy().shape[0]}"
    )
    print(
        f"Bins: RTE:{df[success_bins]['bins_rte'].mean():.3f}+-{df[success_bins]['bins_rte'].std():.3f}, RRE:{df[success_bins]['bins_rre'].mean():.3f}+-{df[success_bins]['bins_rre'].std():.3f}, SR:{success_bins.sum()}/{df['baseline_rte'].to_numpy().shape[0]}"
    )
