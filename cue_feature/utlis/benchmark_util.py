import open3d as o3d
import os
import numpy as np
from os.path import join
import MinkowskiEngine as ME
import copy
from scipy.spatial import cKDTree
from matplotlib import pyplot as plt

import matin
from utlis import trajectory
from utlis import pointcloud
from utlis import files


def run_ransac(xyz0, xyz1, feat0, feat1, voxel_size):
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source=xyz0,
        target=xyz1,
        source_feature=feat0,
        target_feature=feat1,
        mutual_filter=False,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=4,
        checkers=[o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                  o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 0.999),                  # 4000000, 500
    )
    return result_ransac.transformation


def gather_results(results):
    traj = []
    for r in results:
        success = r[0]
        if success:
            traj.append(trajectory.CameraPose([r[1], r[2], r[3]], r[4]))
    return traj


def gen_matching_pair(pts_num):
    matching_pairs = []
    for i in range(pts_num):
        for j in range(i + 1, pts_num):
            matching_pairs.append([i, j, pts_num])
    return matching_pairs


def read_data(feature_path, name):
    data = np.load(os.path.join(feature_path, name + ".npz"))
    xyz = pointcloud.make_open3d_point_cloud(data['xyz'])
    feat = pointcloud.make_open3d_feature_from_numpy(data['feature'])
    return data['points'], xyz, feat


def do_single_pair_matching(feature_path, set_name, m, voxel_size):
    i, j, s = m
    name_i = "%s_cloud_%03d" % (set_name, i)
    name_j = "%s_cloud_%03d" % (set_name, j)
    # logger.info("matching %s %s" % (name_i, name_j))
    points_i, xyz_i, feat_i = read_data(feature_path, name_i)
    points_j, xyz_j, feat_j = read_data(feature_path, name_j)
    if len(xyz_i.points) < len(xyz_j.points):
        trans = run_ransac(xyz_i, xyz_j, feat_i, feat_j, voxel_size)
    else:
        trans = run_ransac(xyz_j, xyz_i, feat_j, feat_i, voxel_size)
        trans = np.linalg.inv(trans)
    ratio = pointcloud.compute_overlap_ratio(xyz_i, xyz_j, trans, voxel_size)
    # logger.info(f"overlapping ratio: {ratio:.3f}")
    if ratio > 0.3:
        return [True, i, j, s, np.linalg.inv(trans)]
    else:
        return [False, i, j, s, np.identity(4)]


def do_single_pair_FMR(feature_path, set_name, traj, voxel_size, tau_1=0.1, tau_2=0.05, num_rand_keypoints=-1):
    trans_gth = np.linalg.inv(traj.pose)
    i = traj.metadata[0]
    j = traj.metadata[1]
    name_i = "%s_cloud_%03d" % (set_name, i)
    name_j = "%s_cloud_%03d" % (set_name, j)

    # coord and feat form a sparse tensor.
    data_i = np.load(join(feature_path, name_i + '.npz'))
    coord_i, points_i, feat_i = data_i['xyz'], data_i['points'], data_i['feature']
    data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
    coord_j, points_j, feat_j = data_j['xyz'], data_j['points'], data_j['feature']

    # use the keypoints in 3DMatch
    if num_rand_keypoints > 0:         # 5000
        # Randomly subsample N points
        Ni, Nj = len(points_i), len(points_j)
        inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)
        inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

        sample_i, sample_j = points_i[inds_i], points_j[inds_j]

        key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))  # randomly downsampled points
        key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

        key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))   # voxelized points
        key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

        inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]              # voxelized points that overlap with randomly downsampled points
        inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

        coord_i, feat_i = coord_i[inds_i], feat_i[inds_i]
        coord_j, feat_j = coord_j[inds_j], feat_j[inds_j]

    coord_i = pointcloud.make_open3d_point_cloud(coord_i)
    coord_j = pointcloud.make_open3d_point_cloud(coord_j)

    hit_ratio = pointcloud.evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth, tau_1)

    # logging.info(f"Hit ratio of {name_i}, {name_j}: {hit_ratio}, {hit_ratio >= tau_2}")
    if hit_ratio >= tau_2:
        return True
    else:
        return False

count = 0
def do_single_pair_ECE(feature_path, snapshot_dir, set_name, traj, voxel_size, tau_1=0.1, tau_2=0.05, num_rand_keypoints=-1):
    trans_gth = np.linalg.inv(traj.pose)
    i = traj.metadata[0]
    j = traj.metadata[1]
    name_i = "%s_cloud_%03d" % (set_name, i)
    name_j = "%s_cloud_%03d" % (set_name, j)

    # coord and feat form a sparse tensor.
    data_i = np.load(join(feature_path, name_i + '.npz'))
    coord_i, points_i, feat_i, sigma2_i = data_i['xyz'], data_i['points'], data_i['feature'], data_i['sigma2']  # 18977
    data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
    coord_j, points_j, feat_j, sigma2_j = data_j['xyz'], data_j['points'], data_j['feature'], data_j['sigma2']

    # use the keypoints in 3DMatch
    if num_rand_keypoints > 0:
        Ni, Nj = len(points_i), len(points_j)
        inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)                  # 5000 Randomly subsample N points
        inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

        sample_i, sample_j = points_i[inds_i], points_j[inds_j]

        key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))  # randomly downsampled points
        key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

        key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))   # downsampled points
        key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

        inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]              # 12167, downsampled points that overlap with randomly downsampled points
        inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

        # --------------------- add uncertainty guidence --------------------- #
        # N_overlap_i = len(inds_i)                                              # ~ 12167
        # unce_overlap_i = sigma2_i[inds_i]                                      # ~ 12167
        # unce_dec_inds_i = np.argsort(unce_overlap_i, 0)                        # ~ 12167
        # inds_i = inds_i[unce_dec_inds_i[:int(0.1 * N_overlap_i)]].flatten()    # ~ 0.9 * 12167

        # N_overlap_j = len(inds_j)
        # unce_overlap_j = sigma2_j[inds_j]
        # unce_dec_inds_j = np.argsort(unce_overlap_j, 0)
        # inds_j = inds_j[unce_dec_inds_j[:int(0.1 * N_overlap_j)]].flatten()
        # --------------------------------- - -------------------------------- #

        coord_i, feat_i, sigma2_i = coord_i[inds_i], feat_i[inds_i], sigma2_i[inds_i]
        coord_j, feat_j, sigma2_j = coord_j[inds_j], feat_j[inds_j], sigma2_j[inds_j]

    coord_i = pointcloud.make_open3d_point_cloud(coord_i)
    coord_j = pointcloud.make_open3d_point_cloud(coord_j)

    # inds_cands, hit_map = evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth, tau_1)  # 12167 <> 12135
    inds_cands, hit_map, mask = evaluate_feature_3dmatch_mutual(coord_i, coord_j, feat_i, feat_j, trans_gth, tau_1) # 12167 <> 12135

    hit_ratio = np.mean(hit_map)    # 12167

    if feat_i.shape[0] < feat_j.shape[0]:
        query_sigma2 = sigma2_i[mask].mean(axis=-1)        # reduce vector to scalar
        cand_sigma2 = sigma2_j.mean(axis=-1)[inds_cands]
    else:
        query_sigma2 = sigma2_j[mask].mean(axis=-1)
        cand_sigma2 = sigma2_i.mean(axis=-1)[inds_cands]

    whole_correct_inds = np.arange(len(query_sigma2))[hit_map]

    # query ECE
    split_inds, bins = get_bins(query_sigma2)              # in increasing order, 12167
    bin_counts_query = np.array([len(x) for x in split_inds])
    bin_hit_ratios_query = np.zeros((len(split_inds)))
    for i, bin_inds in enumerate(split_inds):
        if len(bin_inds) == 0:
            bin_hit_ratios_query[i] = 0
            continue
        bin_correct_inds = np.intersect1d(bin_inds, whole_correct_inds)
        bin_hit_ratios_query[i] = len(bin_correct_inds) / len(bin_inds)

    # pair ECE
    split_inds, bins = get_bins(query_sigma2 + cand_sigma2)                    # in increase order
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
        ax.set_title(f'Avg Hit Ratio = {hit_ratio:.3f}')
        matin.ax_default_style(ax, ratio=0.7, show_grid=True)

        ax = axs[1][0]
        ax.bar(np.arange(len(bin_counts_pair)), bin_counts_pair)
        ax.set_xlabel('Uncertainty Level')
        ax.set_ylabel('Sample Density')
        matin.ax_default_style(ax, ratio=0.65)
        plt.subplots_adjust(hspace=-0.4)

        global count
        single_pc_dir = files.ensure_dir(join(snapshot_dir, 'single_pc'))
        plt.savefig(join(single_pc_dir, f'ece_{count}.png'), bbox_inches='tight')
        count += 1
        plt.close()
    # ----------------------------------- - ---------------------------------- #

    fmr_passed = hit_ratio >= tau_2
    # fmr_passed = bin_hit_ratios_pair[2] >= tau_2

    return [fmr_passed, bin_hit_ratios_query, bin_counts_query, bin_hit_ratios_pair, bin_counts_pair, hit_ratio, query_sigma2 + cand_sigma2, whole_correct_inds]


def do_single_pair_ECE_dual(feature_path, snapshot_dir, set_name, traj, voxel_size, tau_1=0.1, tau_2=0.05, num_rand_keypoints=-1):
    trans_gth = np.linalg.inv(traj.pose)
    i = traj.metadata[0]
    j = traj.metadata[1]
    name_i = "%s_cloud_%03d" % (set_name, i)
    name_j = "%s_cloud_%03d" % (set_name, j)

    # coord and feat form a sparse tensor.
    data_i = np.load(join(feature_path, name_i + '.npz'))
    coord_i, points_i, feat_i, sigma2_i = data_i['xyz'], data_i['points'], data_i['feature'], data_i['sigma2'] # 18977
    data_j = np.load(os.path.join(feature_path, name_j + ".npz"))
    coord_j, points_j, feat_j, sigma2_j = data_j['xyz'], data_j['points'], data_j['feature'], data_j['sigma2']

    # coord and feat form a sparse tensor.
    data_mc_i = np.load(join('logs/HCL_MC_0420_143013/extracted_features', name_i + '.npz'))
    coord_mc_i, points_mc_i, feat_mc_i, sigma2_mc_i = data_mc_i['xyz'], data_mc_i['points'], data_mc_i['feature'], data_mc_i['sigma2'] # 18977
    data_mc_j = np.load(os.path.join('logs/HCL_MC_0420_143013/extracted_features', name_j + ".npz"))
    coord_mc_j, points_mc_j, feat_mc_j, sigma2_mc_j = data_mc_j['xyz'], data_mc_j['points'], data_mc_j['feature'], data_mc_j['sigma2']

    # use the keypoints in 3DMatch
    if num_rand_keypoints > 0:
        Ni, Nj = len(points_i), len(points_j)
        inds_i = np.random.choice(Ni, min(Ni, num_rand_keypoints), replace=False)                  # 5000 Randomly subsample N points
        inds_j = np.random.choice(Nj, min(Nj, num_rand_keypoints), replace=False)

        sample_i, sample_j = points_i[inds_i], points_j[inds_j]

        key_points_i = ME.utils.fnv_hash_vec(np.floor(sample_i / voxel_size))  # randomly downsampled points
        key_points_j = ME.utils.fnv_hash_vec(np.floor(sample_j / voxel_size))

        key_coords_i = ME.utils.fnv_hash_vec(np.floor(coord_i / voxel_size))   # downsampled points
        key_coords_j = ME.utils.fnv_hash_vec(np.floor(coord_j / voxel_size))

        inds_i = np.where(np.isin(key_coords_i, key_points_i))[0]              # 12167, downsampled points that overlap with randomly downsampled points
        inds_j = np.where(np.isin(key_coords_j, key_points_j))[0]

        # --------------------- add uncertainty guidence --------------------- #
        # N_overlap_i = len(inds_i)                                              # ~ 12167
        # unce_overlap_i = sigma2_i[inds_i]                                      # ~ 12167
        # unce_dec_inds_i = np.argsort(unce_overlap_i, 0)                        # ~ 12167
        # inds_i = inds_i[unce_dec_inds_i[:int(0.1 * N_overlap_i)]].flatten()    # ~ 0.9 * 12167

        # N_overlap_j = len(inds_j)
        # unce_overlap_j = sigma2_j[inds_j]
        # unce_dec_inds_j = np.argsort(unce_overlap_j, 0)
        # inds_j = inds_j[unce_dec_inds_j[:int(0.1 * N_overlap_j)]].flatten()
        # --------------------------------- - -------------------------------- #

        coord_i, feat_i, sigma2_i = coord_i[inds_i], feat_i[inds_i], sigma2_i[inds_i]
        coord_j, feat_j, sigma2_j = coord_j[inds_j], feat_j[inds_j], sigma2_j[inds_j]

        coord_mc_i, feat_mc_i, sigma2_mc_i = coord_mc_i[inds_i], feat_mc_i[inds_i], sigma2_mc_i[inds_i]
        coord_mc_j, feat_mc_j, sigma2_mc_j = coord_mc_j[inds_j], feat_mc_j[inds_j], sigma2_mc_j[inds_j]

    coord_i = pointcloud.make_open3d_point_cloud(coord_i)
    coord_j = pointcloud.make_open3d_point_cloud(coord_j)

    inds_cands, hit_map = evaluate_feature_3dmatch(coord_i, coord_j, feat_i, feat_j, trans_gth, tau_1) # 12167 <> 12135
    hit_ratio = np.mean(hit_map)                                                                       # 12167
    fmr_passed = hit_ratio >= tau_2


    # ------------------------- add dual guidence ------------------------ #
    if feat_i.shape[0] < feat_j.shape[0]:
        query_sigma2 = sigma2_i.mean(axis=-1)              # reduce vector to scalar
        cand_sigma2 = sigma2_j.mean(axis=-1)[inds_cands]
    else:
        query_sigma2 = sigma2_j.mean(axis=-1)
        cand_sigma2 = sigma2_i.mean(axis=-1)[inds_cands]
    split_inds, bins = get_bins(query_sigma2 + cand_sigma2)                # in increasing order, 12167
    bin_counts_pair = np.array([len(x) for x in split_inds])

    if feat_i.shape[0] < feat_j.shape[0]:
        query_mc_sigma2 = sigma2_mc_i.mean(axis=-1)          # reduce vector to scalar
        cand_mc_sigma2 = sigma2_mc_j.mean(axis=-1)[inds_cands]
    else:
        query_mc_sigma2 = sigma2_mc_j.mean(axis=-1)
        cand_mc_sigma2 = sigma2_mc_i.mean(axis=-1)[inds_cands]
    split_mc_inds, bins_mc = get_bins(query_mc_sigma2 + cand_mc_sigma2)          # in increasing order, 12167
    bin_counts_pair_mc = np.array([len(x) for x in split_mc_inds])

    whole_correct_inds = np.arange(len(query_sigma2))[hit_map]
    bin_hit_ratios_dual = np.zeros((len(split_inds)))
    split_inds_dual = []
    for i, bin_inds in enumerate(split_inds):
        com_vote = np.intersect1d(split_inds[i], split_mc_inds[i], assume_unique=True)
        split_inds_dual.append(com_vote)
        if len(com_vote) == 0:
            bin_hit_ratios_dual[i] = 0
            continue
        bin_correct_inds = np.intersect1d(com_vote, whole_correct_inds)
        bin_hit_ratios_dual[i] = len(bin_correct_inds) / len(bin_inds)
    bin_counts_pair_dual = np.array([len(x) for x in split_inds_dual])

    # --------------------------------- debug -------------------------------- #
    plt.style.use('ggplot')
    fig, axs = plt.subplots(2, 1, figsize=(5, 10), sharex=True, squeeze=False, dpi=200)
    ax = axs[0][0]
    ax.plot(np.arange(len(bin_hit_ratios_dual)), bin_hit_ratios_dual, marker='o')
    ax.set_ylabel('Hit Ratio')
    ax.set_title(f'Avg Hit Ratio = {hit_ratio:.3f}')
    matin.ax_default_style(ax, ratio=0.7, show_grid=True)

    ax = axs[1][0]
    ax.bar(np.arange(len(bin_counts_pair_dual)), bin_counts_pair_dual)
    ax.set_xlabel('Uncertainty Level')
    ax.set_ylabel('Sample Density')
    matin.ax_default_style(ax, ratio=0.65)
    plt.subplots_adjust(hspace=-0.4)

    global count
    single_pc_dir = files.ensure_dir(join(snapshot_dir, 'single_pc_dual'))
    plt.savefig(join(single_pc_dir, f'ece_{count}.png'), bbox_inches='tight')
    count += 1
    plt.close()
    # ----------------------------------- - ---------------------------------- #

    return [fmr_passed, bin_hit_ratios_dual, bin_counts_pair_dual, bin_hit_ratios_dual, bin_counts_pair_dual, hit_ratio]


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


def evaluate_feature_3dmatch(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh=0.1):
    """
    Return the hit ratio (ratio of inlier correspondences and all correspondences).
    inliear_thresh is the inlier_threshold in meter.
    """
    if len(pcd0.points) < len(pcd1.points):
        inds, hit_map = valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh)
    else:
        inds, hit_map = valid_feat_ratio(pcd1, pcd0, feat1, feat0, np.linalg.inv(trans_gth), inlier_thresh)

    return inds, hit_map


def valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, thresh=0.1):
    pcd0_copy = copy.deepcopy(pcd0)
    pcd0_copy.transform(trans_gth)
    inds = find_nn_cpu(feat0, feat1, return_distance=False)
    dist = np.sqrt(((np.array(pcd0_copy.points) - np.array(pcd1.points)[inds])**2).sum(1))
    return inds, dist < thresh


def evaluate_feature_3dmatch_mutual(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh=0.1):
    """
    Return the hit ratio (ratio of inlier correspondences and all correspondences).
    inliear_thresh is the inlier_threshold in meter.
    """
    if len(pcd0.points) < len(pcd1.points):
        inds, hit_map, mask = valid_feat_ratio_mutual(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh)
    else:
        inds, hit_map, mask = valid_feat_ratio_mutual(pcd1, pcd0, feat1, feat0, np.linalg.inv(trans_gth), inlier_thresh)

    return inds, hit_map, mask


def valid_feat_ratio_mutual(pcd0, pcd1, feat0, feat1, trans_gth, thresh=0.1):
    pcd0_copy = copy.deepcopy(pcd0)
    pcd0_copy.transform(trans_gth)
    inds = find_nn_cpu(feat0, feat1, return_distance=False)
    inds_inv = find_nn_cpu(feat1, feat0, return_distance=False)
    mask = inds_inv[inds] == np.arange(inds.shape[0])
    inds = inds[mask]
    dist = np.sqrt(((np.array(pcd0_copy.points)[mask] - np.array(pcd1.points)[inds])**2).sum(1))
    return inds, dist < thresh, mask


def find_nn_cpu(feat0, feat1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds
