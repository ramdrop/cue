import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


logging_arg = add_argument_group('Logging')
logging_arg.add_argument('--out_dir', type=str, default='logs')

trainer_arg = add_argument_group('Trainer')
trainer_arg.add_argument('--trainer', type=str, default='HCLTrainer')
trainer_arg.add_argument('--phase', type=str, default='')           #  train_sigma |
trainer_arg.add_argument('--save_freq_epoch', type=int, default=1)
trainer_arg.add_argument('--batch_size', type=int, default=4)
trainer_arg.add_argument('--val_batch_size', type=int, default=1)

# Hard negative mining
trainer_arg.add_argument('--use_hard_negative', type=str2bool, default=True)
trainer_arg.add_argument('--hard_negative_sample_ratio', type=int, default=0.05)
trainer_arg.add_argument('--hard_negative_max_num', type=int, default=3000)
trainer_arg.add_argument('--num_pos_per_batch', type=int, default=1024)  # main
trainer_arg.add_argument('--num_hn_samples_per_batch', type=int, default=256)  # sub

# Metric learning loss
trainer_arg.add_argument('--neg_thresh', type=float, default=1.4)
trainer_arg.add_argument('--pos_thresh', type=float, default=0.1)
trainer_arg.add_argument('--bayesian_margin', type=float, default=0)
trainer_arg.add_argument('--neg_weight', type=float, default=1)
trainer_arg.add_argument('--bayesian_sampling_margin', type=float, default=0)
trainer_arg.add_argument('--nb_samples', type=float, default=20)

# Data augmentation
trainer_arg.add_argument('--use_random_scale', type=str2bool, default=False)
trainer_arg.add_argument('--min_scale', type=float, default=0.8)
trainer_arg.add_argument('--max_scale', type=float, default=1.2)
trainer_arg.add_argument('--use_random_rotation', type=str2bool, default=True)
trainer_arg.add_argument('--rotation_range', type=float, default=360)

# Data loader configs
trainer_arg.add_argument('--train_phase', type=str, default="train")
trainer_arg.add_argument('--val_phase', type=str, default="val")
trainer_arg.add_argument('--test_phase', type=str, default="test")

trainer_arg.add_argument('--stat_freq', type=int, default=10)
trainer_arg.add_argument('--test_valid', type=str2bool, default=True)
trainer_arg.add_argument('--val_max_iter', type=int, default=400)
trainer_arg.add_argument('--val_epoch_freq', type=int, default=1)
trainer_arg.add_argument('--positive_pair_search_voxel_size_multiplier', type=float, default=1.5)
trainer_arg.add_argument('--hit_ratio_thresh', type=float, default=0.1)

trainer_arg.add_argument('--eps', type=float, default=1e-7)
trainer_arg.add_argument('--procrustes_weight', type=float, default=1e-4)
trainer_arg.add_argument('--trans_weight', type=float, default=1)

# Triplets
trainer_arg.add_argument('--triplet_num_pos', type=int, default=256)  # hard pos
trainer_arg.add_argument('--triplet_num_hn', type=int, default=512)  # hard neg
trainer_arg.add_argument('--triplet_num_rand', type=int, default=1024)  # ramdon

# dNetwork specific configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--model', type=str, default='DDPNetBN2C')                # DDPNetBN2C | ResUNetBN2C
net_arg.add_argument('--mc_p', type=float, default=0, help='MC Dropout p')
net_arg.add_argument('--model_n_out', type=int, default=32, help='Feature dimension')
net_arg.add_argument('--conv1_kernel_size', type=int, default=5)
net_arg.add_argument('--normalize_feature', type=str2bool, default=True)
net_arg.add_argument('--dist_type', type=str, default='L2')
net_arg.add_argument('--best_val_metric', type=str, default='feat_match_ratio')

# Optimizer arguments
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='SGD')
opt_arg.add_argument('--max_epoch', type=int, default=100)
opt_arg.add_argument('--lr', type=float, default=1e-1)
opt_arg.add_argument('--momentum', type=float, default=0.8)
opt_arg.add_argument('--sgd_momentum', type=float, default=0.9)
opt_arg.add_argument('--sgd_dampening', type=float, default=0.1)
opt_arg.add_argument('--adam_beta1', type=float, default=0.9)
opt_arg.add_argument('--adam_beta2', type=float, default=0.999)
opt_arg.add_argument('--weight_decay', type=float, default=1e-4)
opt_arg.add_argument('--iter_size', type=int, default=1, help='accumulate gradient')
opt_arg.add_argument('--bn_momentum', type=float, default=0.05)
opt_arg.add_argument('--exp_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--icp_cache_path', type=str, default="/home/chrischoy/datasets/FCGF/kitti/icp/")

misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
misc_arg.add_argument('--weights', type=str, default='')
misc_arg.add_argument('--weights_dir', type=str, default=None)
misc_arg.add_argument('--resume', type=str, default=None)
misc_arg.add_argument('--resume_dir', type=str, default=None)
misc_arg.add_argument('--train_num_thread', type=int, default=2)
misc_arg.add_argument('--val_num_thread', type=int, default=1)
misc_arg.add_argument('--test_num_thread', type=int, default=2)
misc_arg.add_argument('--fast_validation', type=str2bool, default=False)
misc_arg.add_argument('--nn_max_n', type=int, default=500, help='The maximum number of features to find nearest neighbors in batch')

# Dataset specific configurations
data_arg = add_argument_group('Data')
data_arg.add_argument('--dataset', type=str, default='ThreeDMatchPairDataset')                     # ThreeDMatchPairDataset | KITTINMPairDataset | ModelNet40PairDataset
data_arg.add_argument('--voxel_size', type=float, default=0.025)
data_arg.add_argument('--threed_match_dir', type=str, default="/LOCAL2/ramdrop/dataset/threedmatch/threedmatch")
data_arg.add_argument('--kitti_root', type=str, default="/LOCAL2/ramdrop/dataset/kitti")
data_arg.add_argument('--kitti_max_time_diff', type=int, default=3, help='max time difference between pairs (non inclusive)')
data_arg.add_argument('--kitti_date', type=str, default='2011_09_26')

# ModelNet40
data_arg.add_argument('--dataset_path', default='/LOCAL2/ramdrop/dataset/ModelNet/ModelNet40_20000', type=str, metavar='PATH', help='path to the input dataset') # like '/path/to/ModelNet40'
data_arg.add_argument('--categoryfile', default='/LOCAL2/ramdrop/dataset/ModelNet/categories/modelnet40_half1.txt', type=str, metavar='PATH', help='path to the categories to be trained')
data_arg.add_argument('--uniformsampling', default=False, type=bool, help='uniform sampling points from the mesh')
data_arg.add_argument('--mag', default=0.8, type=float, metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
data_arg.add_argument('--perturbations', default='/LOCAL2/ramdrop/dataset/ModelNet/categories/pert_test.csv', type=str, metavar='PATH', help='path to the perturbation file')

def get_options():
    args = parser.parse_args()
    return args
