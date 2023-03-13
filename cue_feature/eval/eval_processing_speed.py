#%%
import models
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import pickle
import time
import numpy as np
np.set_printoptions(precision=6, suppress=True)

device = torch.device('cuda')


def get_model(model_path):
    checkpoint = torch.load(model_path)
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
    # print(f"loading weight from epoch {checkpoint['epoch']}, best val FMR={checkpoint['best_val']}")
    model.eval()

    model = model.to(device)
    return model


with open('logs/results/sinput.pickle', 'rb') as handle:
    coords = pickle.load(handle)
    feats = pickle.load(handle)
sinput = ME.SparseTensor(feats, coordinates=coords, device=device, requires_grad=True)

model0 = get_model('logs/HCL_0419_161400/best_val_checkpoint.pth')
model1 = get_model('logs/BTL_0804_095744/best_val_checkpoint.pth')
model2 = get_model('logs/MBTL_0805_024054/best_val_checkpoint.pth')

ts = []
with torch.no_grad():
    for i in range(10):
        t0 = time.time()
        for j in range(30):
            _ = model0(sinput)
        t1 = time.time()
        ts.append((t1 - t0) / 30.0)
ts = np.array(ts)
print(ts.mean(), ts.var())

ts = []
with torch.no_grad():
    for i in range(10):
        t0 = time.time()
        for j in range(30):
            _ = model1(sinput)
        t1 = time.time()
        ts.append((t1 - t0) / 30.0)
ts = np.array(ts)
print(ts.mean(), ts.var())

ts = []
with torch.no_grad():
    for i in range(10):
        t0 = time.time()
        for j in range(30):
            _ = model2(sinput)
        t1 = time.time()
        ts.append((t1 - t0) / 30.0)
ts = np.array(ts)
print(ts.mean(), ts.var())

#%%
