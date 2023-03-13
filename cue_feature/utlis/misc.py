import torch
import numpy as np
import MinkowskiEngine as ME


def _hash(arr, M):
    if isinstance(arr, np.ndarray):
        N, D = arr.shape
    else:
        N, D = len(arr[0]), len(arr)

    hash_vec = np.zeros(N, dtype=np.int64)
    for d in range(D):
        if isinstance(arr, np.ndarray):
            hash_vec += arr[:, d] * M**d
        else:
            hash_vec += arr[d] * M**d
    return hash_vec


def extract_features(model, xyz, rgb=None, normal=None, voxel_size=0.05, device=None, skip_check=False, is_eval=True, repeat_n=0):
    """
    xyz is a N x 3 matrix
    rgb is a N x 3 matrix and all color must range from [0, 1] or None
    normal is a N x 3 matrix and all normal range from [-1, 1] or None

    if both rgb and normal are None, we use Nx1 one vector as an input
    if device is None, it tries to use gpu by default
    if skip_check is True, skip rigorous checks to speed up

    model = model.to(device)
    xyz, feats = extract_features(model, xyz)
    """

    if is_eval:
        model.eval()

    if not skip_check:
        assert xyz.shape[1] == 3

        N = xyz.shape[0]
        if rgb is not None:
            assert N == len(rgb)
            assert rgb.shape[1] == 3
            if np.any(rgb > 1):
                raise ValueError('Invalid color. Color must range from [0, 1]')

        if normal is not None:
            assert N == len(normal)
            assert normal.shape[1] == 3
            if np.any(normal > 1):
                raise ValueError('Invalid normal. Normal must range from [-1, 1]')

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    feats = []
    if rgb is not None:
        # [0, 1]
        feats.append(rgb - 0.5)

    if normal is not None:
        # [-1, 1]
        feats.append(normal / 2)

    if rgb is None and normal is None:
        feats.append(np.ones((len(xyz), 1)))

    feats = np.hstack(feats)

    # voxelize xyz and feats
    coords = np.floor(xyz / voxel_size)                    # voxel_size=0.05
    coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
    xyz_down = xyz[inds]               # downsampled points (in each voxel)

    # convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    feats = feats[inds]

    # convert data type
    # coords = torch.tensor(coords, dtype=torch.int32)
    coords = coords.type(torch.int32)
    feats = torch.tensor(feats, dtype=torch.float32)

    sinput = ME.SparseTensor(feats, coordinates=coords, device=device, requires_grad=True)
    # import pickle
    # with open('sinput.pickle', 'wb') as handle:
    #     pickle.dump(coords, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     pickle.dump(feats, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # import pickle
    # with open('sinput.pickle', 'rb') as handle:
    #     coords = pickle.load(handle)
    #     feats = pickle.load(handle)
    # sinput = ME.SparseTensor(feats, coordinates=coords, device=device, requires_grad=True)

    if repeat_n == 0:
        soutput = model(sinput)
        if 'ResUNet' in model._get_name():                 # determinsitc model
            mu = soutput.F
            sigma2 = 0
        elif 'DDP' in model._get_name() or 'MG' in model._get_name():                   # probablistic model
            mu = soutput[0].F
            sigma2 = soutput[1].F
    else:                                                  # MC Dropout model
        mus = torch.zeros((repeat_n, len(sinput), 32)).to(device)
        model = model_switch(model, bn='freeze', dropout='enable')  # == model.eval() + dropout
        for idx in range(repeat_n):
            soutput = model(sinput)
            if 'ResUNet' in model._get_name():
                mus[idx] = soutput.F
            elif 'DDP' in model._get_name():
                mus[idx] = soutput[0].F
        # mu, sigma2 = torch.var_mean(mus, dim=0)
        sigma2, mu  = torch.var_mean(mus, dim=0)

        # ----------------- override mu with dropout disabled ---------------- #
        model = model_switch(model, bn='freeze', dropout='disable') # == model.eval()
        if 'ResUNet' in model._get_name():                 # determinsitc model
            mu = soutput.F
        elif 'DDP' in model._get_name():                   # probablistic model
            mu = soutput[0].F

    return xyz_down, mu, sigma2

def model_switch(model, bn='freeze', dropout='enable'):
    if bn == 'update':
        model.train()
        if dropout == 'enable':
            pass
        elif dropout == 'disable':
            for module in model.modules():
                if isinstance(module, ME.MinkowskiConvolution):
                    module.training = False
    elif bn == 'freeze':
        model.eval()
        if dropout == 'enable':
            for module in model.modules():
                if isinstance(module, ME.MinkowskiConvolution):
                    module.training = True
        elif dropout == 'disable':
            pass
    return model

def find_correct_correspondence(pos_pairs, pred_pairs, hash_seed=None, len_batch=None):
    assert len(pos_pairs) == len(pred_pairs)
    if hash_seed is None:
        assert len(len_batch) == len(pos_pairs)

    corrects = []
    for i, pos_pred in enumerate(zip(pos_pairs, pred_pairs)):
        pos_pair, pred_pair = pos_pred
        if isinstance(pos_pair, torch.Tensor):
            pos_pair = pos_pair.numpy()
        if isinstance(pred_pair, torch.Tensor):
            pred_pair = pred_pair.numpy()

        if hash_seed is None:
            N0, N1 = len_batch[i]
            _hash_seed = max(N0, N1)
        else:
            _hash_seed = hash_seed

        pos_keys = _hash(pos_pair, _hash_seed)
        pred_keys = _hash(pred_pair, _hash_seed)

        corrects.append(np.isin(pred_keys, pos_keys, assume_unique=False))

    return np.hstack(corrects)
