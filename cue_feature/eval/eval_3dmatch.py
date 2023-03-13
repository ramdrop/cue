"""
A collection of unrefactored functions.
"""
import os
import sys
sys.path.append('.')
import numpy as np
import argparse
import open3d as o3d
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from tqdm import tqdm
from os.path import join, dirname, abspath
import pickle
import shutil
from datetime import datetime

import matin
import models
from utlis import timer
from utlis import misc
from utlis import files
from utlis import trajectory
from utlis import pointcloud
from utlis import benchmark_util

global logging
np.set_printoptions(precision=3, suppress=True)


def a_extract_features(model, config, source_path, target_path, voxel_size, device):

    folders = files.get_folder_list(source_path)
    assert len(folders) > 0, f"Could not find 3DMatch folders under {source_path}"
    list_file = os.path.join(target_path, "list.txt")
    f = open(list_file, "w")
    _timer, tmeter = timer.Timer(), timer.AverageMeter()
    num_feat = 0
    model.eval()

    bar1 = tqdm(folders, colour='blue', unit='batch', leave=False)
    for folder in bar1:
        bar1.set_description(f'folder: {os.path.basename(folder)}')
        if 'evaluation' in folder:
            continue
        pcd_files = files.get_file_list(folder, ".ply")
        folder_name = os.path.basename(folder)
        f.write(f"{folder_name} {len(pcd_files)}\n")
        bar2 = tqdm(pcd_files, colour='blue', unit='batch', leave=False)
        for i, pcd_file in enumerate(bar2):
            pcd = o3d.io.read_point_cloud(pcd_file)

            _timer.tic()
            xyz_down, mu, sigma2 = misc.extract_features(
                model,
                xyz=np.array(pcd.points),
                rgb=None,
                normal=None,
                voxel_size=voxel_size,
                device=device,
                skip_check=True,
                repeat_n=40 if 'mc_p' in config and config.mc_p != 0 else 0,
            )
            t = _timer.toc()

            if i > 0:
                tmeter.update(t)
                num_feat += len(xyz_down)
                bar2.set_description(f'Avg Time/PCS: {tmeter.avg:.4f}, FPS: {num_feat / tmeter.sum:.4f}')

            np.savez_compressed(join(target_path, f'{folder_name}_cloud_{i:03d}'), \
                    points=np.array(pcd.points), \
                    xyz=xyz_down, \
                    feature=mu.detach().cpu().numpy(),\
                    sigma2=sigma2.detach().cpu().numpy())                                                          # points: raw points, xyz: downsampled points, embs: point embeddings
    # print(f'Avg Time/PCS: {tmeter.avg:.4f}, FPS: {num_feat / tmeter.sum:.4f}')
    f.close()


def b_evaluate_FMR(source_path, feature_path, voxel_size, num_rand_keypoints=-1):
    with open(os.path.join(feature_path, "list.txt")) as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]

    assert len(sets) > 0, "Empty list file. Makesure to run the feature extraction first with --do_extract_feature."

    tau_1 = 0.1                        # 10cm
    tau_2 = 0.05                       # 5% inlier
    logging.info(f'{tau_1}, {tau_2}\n')

    recall = []
    bar = tqdm(sets, colour='blue', unit='batch', leave=False)
    for s in bar:
        bar.set_description(f'{s[0]}')
        set_name = s[0]
        traj = trajectory.read_trajectory(os.path.join(source_path, set_name + "-evaluation/gt.log"))
        assert len(traj) > 0, "Empty trajectory file"
        results = []
        for i in tqdm(range(len(traj)), leave=False):
            results.append(benchmark_util.do_single_pair_FMR(feature_path, set_name, traj[i], voxel_size, tau_1, tau_2, num_rand_keypoints))
        mean_recall = np.array(results).mean()
        std_recall = np.array(results).std()
        recall.append([set_name, mean_recall, std_recall])

    for r in recall:
        logging.info(f'{r[1]:.3f} +- {r[2]:.3f} | {r[0]}')
    scene_r = np.array([r[1] for r in recall])
    logging.info(f'{scene_r.mean():.3f} +- {scene_r.std():.3f} | Average FMR')


def c_registration(feature_path, voxel_size):
    """
    Gather .log files produced in --target folder and run this Matlab script
    https://github.com/andyzeng/3dmatch-toolbox#geometric-registration-benchmark
    (see Geometric Registration Benchmark section in
    http://3dmatch.cs.princeton.edu/)
    Matlab script: https://github.com/qianyizh/ElasticReconstruction
    """
    # List file from the extract_features_batch function
    with open(os.path.join(feature_path, "list.txt")) as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]
    for s in tqdm(sets):
        set_name = s[0]
        pts_num = int(s[1])
        matching_pairs = benchmark_util.gen_matching_pair(pts_num)
        results = []
        bar = tqdm(matching_pairs, leave=False)
        for m in bar:
            bar.set_description(f'{feature_path}/{set_name}.log')
            results.append(benchmark_util.do_single_pair_matching(feature_path, set_name, m, voxel_size))
        traj = benchmark_util.gather_results(results)
        # logging.info(f"Writing the trajectory to {feature_path}/{set_name}.log")
        trajectory.write_trajectory(traj, "%s.log" % (os.path.join(feature_path, set_name)))


def d_evaluate_ECE(source_path, feature_path, snapshot_dir, voxel_size, num_rand_keypoints=-1):
    with open(os.path.join(feature_path, "list.txt")) as f:
        sets = f.readlines()
        sets = [x.strip().split() for x in sets]

    assert len(sets) > 0, "Empty list file. Makesure to run the feature extraction first with --do_extract_feature."

    tau_1 = 0.1                        # 10cm
    tau_2 = 0.05                       # 5% inlier
    logging.info(f'{tau_1}, {tau_2}\n')

    fmr_sets = []
    query_ECE_sets = []
    query_ECE_counts_sets = []
    pair_ECE_sets = []
    pair_ECE_counts_sets = []
    hit_ratios_sets = []
    sigma_sets = []
    correct_inds_sets = []

    # ------------------------------- if debug ------------------------------- #
    # sets = sets[:1]

    bar = tqdm(sets, colour='blue', unit='batch', leave=False)
    for s in bar:
        bar.set_description(f'{s[0]}')
        set_name = s[0]
        traj = trajectory.read_trajectory(os.path.join(source_path, set_name + "-evaluation/gt.log"))
        assert len(traj) > 0, "Empty trajectory file"
        results = []
        for i in tqdm(range(len(traj)), leave=False):
            results.append(benchmark_util.do_single_pair_ECE(feature_path, snapshot_dir, set_name, traj[i], voxel_size, tau_1, tau_2, num_rand_keypoints))
        fmr_passed, query_ECE, query_counts, pair_ECE, pair_counts, hit_ratios, sigma, correct_inds = \
            [x[0] for x in results], \
            np.array([x[1] for x in results]), \
            np.array([x[2] for x in results]), \
            np.array([x[3] for x in results]), \
            np.array([x[4] for x in results]), \
            np.array([x[5] for x in results]), \
            [x[6] for x in results],    \
            [x[7] for x in results],

        fmr_sets.append([set_name, np.array(fmr_passed).mean(), np.array(fmr_passed).std()])

        query_ECE_sets.append([set_name, query_ECE])
        query_ECE_counts_sets.append([set_name, query_counts])
        pair_ECE_sets.append([set_name, pair_ECE])
        pair_ECE_counts_sets.append([set_name, pair_counts])
        hit_ratios_sets.append([set_name, hit_ratios])
        sigma_sets.append([set_name, sigma])
        correct_inds_sets.append([set_name, correct_inds])

    for r in fmr_sets:
        logging.info(f'{r[1]:.3f} +- {r[2]:.3f} | {r[0]}')
    fmr_sets_avg = np.array([r[1] for r in fmr_sets])
    logging.info(f'{fmr_sets_avg.mean():.3f} +- {fmr_sets_avg.std():.3f} | Average FMR\n')

    with open(join(snapshot_dir, 'ece_results.pickle'), 'wb') as handle:
        pickle.dump(fmr_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(query_ECE_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(query_ECE_counts_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pair_ECE_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(pair_ECE_counts_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(hit_ratios_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(sigma_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(correct_inds_sets, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    def str2bool(v):
        return v.lower() in ('true', '1')
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='logs/MBTL_0805_024054/best_val_checkpoint.pth', type=str, help='path to latest checkpoint')
    parser.add_argument('--source', default='dbs/threedmatch/testbench', type=str, help='path to 3dmatch test dataset')
    parser.add_argument('--target', default=None, type=str, help='path to produce generated data')
    parser.add_argument('--voxel_size', default=0.05, type=float, help='voxel size to preprocess point cloud')
    parser.add_argument('--num_rand_keypoints', type=int, default=5000, help='Number of random keypoints for each scene')
    parser.add_argument('--with_cuda', type=str2bool, default=True)
    parser.add_argument('--extract_features', type=str2bool, default=0)
    parser.add_argument('--evaluate_FMR', type=str2bool, default=0)
    parser.add_argument('--evaluate_registration', type=str2bool, default=0)
    parser.add_argument('--evaluate_ECE', type=str2bool, default=1)
    args = parser.parse_args()

    torch.cuda.set_device(matin.schedule_device())
    device = torch.device('cuda' if args.with_cuda else 'cpu')

    # ---------------------- save evaluation parameters ---------------------- #
    args.target = files.ensure_dir(join(dirname(args.model), 'extracted_features'))
    snapshot_dir = join(dirname(args.model), f"eval_{datetime.now().strftime('%m%d_%H%M%S')}")
    # snapshot_dir = files.inc_dirname(dirname(args.model), 'eval_*')
    snapshot_dir = files.ensure_dir(snapshot_dir)       # join(dirname(args.model), 'eval_0')

    logging = matin.ln(__name__, tofile=join(snapshot_dir, 'benchmark_results.log')).get_logger()
    shutil.copy('utlis/benchmark_util.py', snapshot_dir)

    if args.extract_features:
        assert args.model is not None
        assert args.source is not None
        assert args.target is not None

        files.ensure_dir(args.target)
        checkpoint = torch.load(args.model)
        config = checkpoint['config']

        num_feats = 1
        Model = models.load_model(config.model)
        model = Model(num_feats, config.model_n_out, bn_momentum=0.05, normalize_feature=config.normalize_feature, conv1_kernel_size=config.conv1_kernel_size, D=3)

        # ------------------------- apply MC Dropout ------------------------- #
        if 'mc_p' in config and config.mc_p != 0:
            def dropout_hook_wrapper(module, sinput, soutput):
                input = soutput.F
                output = F.dropout(input, p=config.mc_p, training=module.training)   # force training state for Dropout layers
                soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
                return soutput_new
            for module in model.modules():
                if isinstance(module, ME.MinkowskiConvolution):
                    module.register_forward_hook(dropout_hook_wrapper)

        model.load_state_dict(checkpoint['state_dict'])
        logging.info(f"loading weight from epoch {checkpoint['epoch']}, best val FMR={checkpoint['best_val']}")
        model.eval()

        model = model.to(device)

        with torch.no_grad():
            a_extract_features(model, config, args.source, args.target, config.voxel_size, device)

    if args.evaluate_FMR:
        assert (args.target is not None)
        with torch.no_grad():
            b_evaluate_FMR(args.source, args.target, args.voxel_size, args.num_rand_keypoints)

    if args.evaluate_registration:
        assert (args.target is not None)
        with torch.no_grad():
            c_registration(args.target, args.voxel_size)

    if args.evaluate_ECE:
        assert (args.target is not None)
        with torch.no_grad():
            d_evaluate_ECE(args.source, args.target, snapshot_dir, args.voxel_size, args.num_rand_keypoints)
