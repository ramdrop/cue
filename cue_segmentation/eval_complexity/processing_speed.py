#%%
import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MEF
import pickle
import time
import numpy as np
from tqdm import tqdm
import gin
import sys
sys.path.append('.')
from src.mbox import com
from src.models import get_model
np.set_printoptions(precision=3, suppress=True)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pointer", type=str, default='mink')           # cue | lrmg | entropy | aleatoric | mc | dul | rul
args = parser.parse_args()


POINTER = args.pointer
CANDIDATE = {
    'mink': ['Res16UNet34C', 'logs_scannet/2mink_0623_203059/epoch=659-step=99660.ckpt'],
    'cue': ['Res16UNet34CProb', 'logs_scannet/2minkprob_0623_213340/epoch=582-step=88033.ckpt'],
    'cueplus': ['Res16UNet34CProbMG', 'logs_scannet/minkprobmg_0805_084511/epoch=612-step=92563.ckpt'],
    'se': ['Res16UNet34C', 'logs_scannet/2mink_0623_203059/epoch=659-step=99660.ckpt'],
    'au': ['Res16UNet34CAleatoric', 'logs_scannet/minkaleatoric_0803_080321/epoch=622-step=94073.ckpt'],
    'mcd': ['Res16UNet34CMC', 'logs_scannet/minkmc_0817_065324/epoch=610-step=92261.ckpt'],
    'dul': ['Res16UNet34CDUL', 'logs_scannet/minkdul_1230_095824/epoch=554-step=83805.ckpt'],
    'rul': ['Res16UNet34CRUL', 'logs_scannet/minkrul_1230_191051/epoch=547-step=82748.ckpt'],
}

gin.parse_config_file(f'config/scannet/minimum_model/eval_{POINTER}.gin')

model_name = CANDIDATE[POINTER][0]        # Res16UNet34C|Res16UNet34CProb
checkpoint_path=CANDIDATE[POINTER][1]

model = get_model(model_name)()

if 'MC' in model_name:
    # p_mc = gin.query_parameter('Res16UNet34CMC.p')
    p_mc = 0.05
    print(f'applying MC dropout p={p_mc} after every conv layer..')
    def dropout_hook_wrapper_another(module, sinput, soutput):
        soutput = MEF.dropout(soutput, p=0.2, training=module.training)
        return soutput
    def dropout_hook_wrapper(module, sinput, soutput):
        input = soutput.F
        output = F.dropout(input, p=p_mc, training=module.training)
        soutput_new = ME.SparseTensor(output, coordinate_map_key=soutput.coordinate_map_key, coordinate_manager=soutput.coordinate_manager)
        return soutput_new
    for module in model.modules():
        if isinstance(module, ME.MinkowskiConvolution):
            module.register_forward_hook(dropout_hook_wrapper)


torch.cuda.set_device(1)
device = torch.device("cuda")

def remove_prefix(k, prefix):
    return k[len(prefix):] if k.startswith(prefix) else k
ckpt = torch.load(checkpoint_path)
state_dict = {remove_prefix(k, "model."): v for k, v in ckpt["state_dict"].items()}
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()


num_of_para = 0
for module in model.modules():
    if isinstance(module, ME.MinkowskiConvolution):
        para = 1
        for k in module.kernel.shape:
            para *= k
        num_of_para += para

with open('eval_complexity/sinput.pickle', 'rb') as handle:
    batch = pickle.load(handle)

num_samples = 5
num_forward = 1 if POINTER == 'mcd' else 10
ts = []
with torch.no_grad():
    for i in tqdm(range(num_samples), leave=False):
        t0 = time.time()
        for j in range(num_forward):
            if POINTER == 'mcd':
                num_repeat = 40
                logits_multiple = torch.zeros((num_repeat, len(batch["features"]), 20)).to(device)
                for ind in range(num_repeat):
                    in_data_ = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                    logits_multiple[ind] = model(in_data_)                     # ([Nr, num_cls])
                sigma, logits = torch.var_mean(logits_multiple, dim=0)
                pred_dense = logits.argmax(dim=1, keepdim=False)
                # y = F.one_hot(torch.from_numpy(pred_dense), num_classes=20).numpy()                # [Nr, 20]
                y = F.one_hot(pred_dense, num_classes=20)                # [Nr, 20]
                q = (sigma - sigma.min(axis=1, keepdims=True)[0]) / (sigma.max(axis=1, keepdims=True)[0] - sigma.min(axis=1, keepdims=True)[0])# [Nr, 20]
                sigma = y * (1 - 0.5 * q) + (1 - y) * 0.5 * q
                sigma = sigma.sum(axis=1)
            else:
                in_data = ME.TensorField(features=batch["features"], coordinates=batch["coordinates"], quantization_mode=model.QMODE, device=device)
                soutput = model(in_data)
                if POINTER == 'se':
                    seg_prob = torch.nn.functional.softmax(soutput, dim=1)
                    entropy = (seg_prob * torch.log(seg_prob)).sum(dim=-1, keepdim=True) / torch.log(torch.tensor(seg_prob.shape[-1]))
                elif POINTER == 'au':
                    logits, sigma = soutput    # ([Nr, num_cls])
                    pred_dense = logits.argmax(dim=1, keepdim=False)
                    # y = F.one_hot(torch.from_numpy(pred_dense), num_classes=20).numpy()                # [Nr, 20]
                    y = F.one_hot(pred_dense, num_classes=20)                # [Nr, 20]
                    q = (sigma - sigma.min(axis=1, keepdims=True)[0]) / (sigma.max(axis=1, keepdims=True)[0] - sigma.min(axis=1, keepdims=True)[0])# [Nr, 20]
                    sigma = y * (1 - 0.5 * q) + (1 - y) * 0.5 * q
                    sigma = sigma.sum(axis=1)
        t1 = time.time()
        ts.append((t1 - t0) / num_forward)
ts = np.array(ts)

print(f"{POINTER}, {num_samples}, {num_forward} | paras:{num_of_para/1e6:.2f}M, {ts.mean():.3f} +- {ts.var():.3f}")
