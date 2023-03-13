import copy
import numpy as np
import math
import time
from scipy.spatial import cKDTree
import open3d as o3d
from utlis.metrics import pdist
import torch

def find_nn_cpu(feat0, feat1, return_distance=False):
    feat1tree = cKDTree(feat1)
    dists, nn_inds = feat1tree.query(feat0, k=1, n_jobs=-1)
    if return_distance:
        return nn_inds, dists
    else:
        return nn_inds


def find_nn_gpu(F0, F1, nn_max_n=-1, return_distance=False, dist_type='SquareL2'):
    # Too much memory if F0 or F1 large. Divide the F0
    if nn_max_n > 1:
        N = len(F0)
        C = int(np.ceil(N / nn_max_n))
        stride = nn_max_n
        dists, inds = [], []
        for i in range(C):
            dist = pdist(F0[i * stride:(i + 1) * stride], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        if C * stride < N:
            dist = pdist(F0[C * stride:], F1, dist_type=dist_type)
            min_dist, ind = dist.min(dim=1)
            dists.append(min_dist.detach().unsqueeze(1).cpu())
            inds.append(ind.cpu())

        dists = torch.cat(dists)
        inds = torch.cat(inds)
        assert len(inds) == N
    else:
        dist = pdist(F0, F1, dist_type=dist_type)
        min_dist, inds = dist.min(dim=1)
        dists = min_dist.detach().unsqueeze(1).cpu()
        inds = inds.cpu()
    if return_distance:
        return inds, dists
    else:
        return inds


def make_open3d_point_cloud(xyz, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    return pcd


def make_open3d_feature(data, dim, npts):
    feature = o3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    feature.data = data.cpu().numpy().astype('d').transpose()
    return feature


def make_open3d_feature_from_numpy(data):
    assert isinstance(data, np.ndarray)
    assert data.ndim == 2

    feature = o3d.pipelines.registration.Feature()
    feature.resize(data.shape[1], data.shape[0])
    feature.data = data.astype('d').transpose()
    return feature


def prepare_pointcloud(filename, voxel_size):
    pcd = o3d.io.read_point_cloud(filename)
    T = get_random_transformation(pcd)
    pcd.transform(T)
    pcd_down = pcd.voxel_down_sample(voxel_size)
    return pcd_down, T


def compute_overlap_ratio(pcd0, pcd1, trans, voxel_size):
    pcd0_down = pcd0.voxel_down_sample(voxel_size)
    pcd1_down = pcd1.voxel_down_sample(voxel_size)
    matching01 = get_matching_indices_original(pcd0_down, pcd1_down, trans, voxel_size, 1)
    matching10 = get_matching_indices_original(pcd1_down, pcd0_down, np.linalg.inv(trans), voxel_size, 1)
    overlap0 = len(matching01) / len(pcd0_down.points)
    overlap1 = len(matching10) / len(pcd1_down.points)
    return max(overlap0, overlap1)


def get_matching_indices_original(source, target, trans, search_voxel_size, K=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    match_inds = []
    for i, point in enumerate(source_copy.points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            match_inds.append((i, j))
    return match_inds



def get_matching_indices(source, target, trans, search_voxel_size, K=None, writer=None):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)

    # from PIL import ImageColor
    # from torch.utils.tensorboard import SummaryWriter

    # plate = lambda x: np.array([ImageColor.getcolor(x, "RGB")])
    # logdir = "/LOCAL2/ramdrop/github/point_registration/DPR/logs_modelnet"
    # gwriter = SummaryWriter(logdir)

    # v0 = np.array(source_copy.points)
    # c0 = np.copy(v0)
    # c0[:, :] = plate('#f56c42')
    # v1 = np.array(target.points)
    # c1 = np.copy(v1)
    # c1[:, :] = plate('#02678e')
    # v2 = np.vstack((v0, v1))
    # c2 = np.vstack((c0, c1))

    # v3 = np.array(source.points)
    # c3 = np.copy(v3)
    # c3[:, :] = plate('#f56c42')
    # v4 = np.array(target.points)
    # c4 = np.copy(v4)
    # c4[:, :] = plate('#02678e')
    # v5 = np.vstack((v3, v4))
    # c5 = np.vstack((c3, c4))

    # gwriter.add_mesh('aligned/v0', vertices=torch.tensor(v0).unsqueeze(0), colors=torch.tensor(c0, dtype=torch.int).unsqueeze(0), global_step=0)
    # gwriter.add_mesh('aligned/v1', vertices=torch.tensor(v1).unsqueeze(0), colors=torch.tensor(c1, dtype=torch.int).unsqueeze(0), global_step=0)
    # gwriter.add_mesh('aligned/v2', vertices=torch.tensor(v2).unsqueeze(0), colors=torch.tensor(c2, dtype=torch.int).unsqueeze(0), global_step=0)
    # gwriter.add_mesh('perturbed/v3', vertices=torch.tensor(v3).unsqueeze(0), colors=torch.tensor(c3, dtype=torch.int).unsqueeze(0), global_step=0)
    # gwriter.add_mesh('perturbed/v4', vertices=torch.tensor(v4).unsqueeze(0), colors=torch.tensor(c4, dtype=torch.int).unsqueeze(0), global_step=0)
    # gwriter.add_mesh('perturbed/v5', vertices=torch.tensor(v5).unsqueeze(0), colors=torch.tensor(c5, dtype=torch.int).unsqueeze(0), global_step=0)
    # gwriter.flush()
    # gwriter.close()
    match_inds = []
    pos0_inds = []
    pos_ind0_pool = np.random.choice(len(source_copy.points), int(0.3*len(source_copy.points)), replace=False)
    for i in pos_ind0_pool:
        [_, idx, _] = pcd_tree.search_radius_vector_3d(source_copy.points[i], search_voxel_size)
        if K is not None:
            idx = idx[:K]
        if len(idx) > 0:
            pos0_inds.append(i)
        for j in idx:
            match_inds.append((i, j))
        if len(pos0_inds) > ((1024+256) - 1):
            break
    return match_inds

def evaluate_feature(pcd0, pcd1, feat0, feat1, trans_gth, search_voxel_size):
    match_inds = get_matching_indices(pcd0, pcd1, trans_gth, search_voxel_size)
    pcd_tree = o3d.geometry.KDTreeFlann(feat1)
    dist = []
    for ind in match_inds:
        k, idx, _ = pcd_tree.search_knn_vector_xd(feat0.data[:, ind[0]], 1)
        dist.append(
            np.clip(
                np.power(pcd1.points[ind[1]] - pcd1.points[idx[0]], 2),
                a_min=0.0,
                a_max=1.0))
    return np.mean(dist)


def valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, thresh=0.1):
    pcd0_copy = copy.deepcopy(pcd0)
    pcd0_copy.transform(trans_gth)
    inds = find_nn_cpu(feat0, feat1, return_distance=False)
    dist = np.sqrt(((np.array(pcd0_copy.points) - np.array(pcd1.points)[inds])**2).sum(1))
    return np.mean(dist < thresh)


def evaluate_feature_3dmatch(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh=0.1):
    """ 
    Return the hit ratio (ratio of inlier correspondences and all correspondences).
    inliear_thresh is the inlier_threshold in meter.
    """
    if len(pcd0.points) < len(pcd1.points):
        hit = valid_feat_ratio(pcd0, pcd1, feat0, feat1, trans_gth, inlier_thresh)
    else:
        hit = valid_feat_ratio(pcd1, pcd0, feat1, feat0, np.linalg.inv(trans_gth), inlier_thresh)
    return hit


def get_matching_matrix(source, target, trans, voxel_size, debug_mode):
    source_copy = copy.deepcopy(source)
    target_copy = copy.deepcopy(target)
    source_copy.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(target_copy)
    matching_matrix = np.zeros((len(source_copy.points), len(target_copy.points)))

    for i, point in enumerate(source_copy.points):
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point, voxel_size * 1.5)
        if k >= 1:
            matching_matrix[i, idx[0]] = 1  # TODO: only the cloest?

    return matching_matrix


def get_random_transformation(pcd_input):

    def rot_x(x):
        out = np.zeros((3, 3))
        c = math.cos(x)
        s = math.sin(x)
        out[0, 0] = 1
        out[1, 1] = c
        out[1, 2] = -s
        out[2, 1] = s
        out[2, 2] = c
        return out

    def rot_y(x):
        out = np.zeros((3, 3))
        c = math.cos(x)
        s = math.sin(x)
        out[0, 0] = c
        out[0, 2] = s
        out[1, 1] = 1
        out[2, 0] = -s
        out[2, 2] = c
        return out

    def rot_z(x):
        out = np.zeros((3, 3))
        c = math.cos(x)
        s = math.sin(x)
        out[0, 0] = c
        out[0, 1] = -s
        out[1, 0] = s
        out[1, 1] = c
        out[2, 2] = 1
        return out

    pcd_output = copy.deepcopy(pcd_input)
    mean = np.mean(np.asarray(pcd_output.points), axis=0).transpose()
    xyz = np.random.uniform(0, 2 * math.pi, 3)
    R = np.dot(np.dot(rot_x(xyz[0]), rot_y(xyz[1])), rot_z(xyz[2]))
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = np.dot(-R, mean)
    T[3, 3] = 1
    return T
