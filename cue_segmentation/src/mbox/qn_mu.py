#%%
import numpy as np
np.set_printoptions(precision=3, suppress=True)
from glob import glob
from os.path import exists, join, dirname
import os
import sys
sys.path.append('.')
from src.mbox import com
from tqdm import tqdm
import torchmetrics
import torch
import pandas as pd


# eval_dir = 'logs_scannet/2mink_0623_203059/eval_0712_140547'    # ENT(Mask), CE
# eval_dir = 'logs_scannet/minkprob_0705_065734/eval_0712_124534'               # ENT(Cluster), CLS=21, CE
# eval_dir = 'logs_scannet/minkprob_0714_023141/eval_0715_065601'               # ENT(Cluster), CLS=21, CE+BTL
# eval_dir = 'logs_scannet/2minkprob_0623_213340/eval_0712_132839'    # CUE, CE+BTL


method = 'mc_005'    # 'cue' | 'lrmg' | 'entropy' | 'aleatoric' | 'mc'

NUM_OF_CLS = 21 if method == 'entropy21' else 20

eval_dir_dic = com.get_eval_dic()
eval_dir = eval_dir_dic[method]
print(f'evaluating {method}: {eval_dir} ..')


filelist = glob(join(eval_dir, 'meta', '*_label_dense.npy'))
pcd_dir = join(eval_dir, 'pcd')
if not exists(pcd_dir):
    os.makedirs(pcd_dir)
ece_dir = join(eval_dir, 'ece')
if not exists(ece_dir):
    os.makedirs(ece_dir)

filelist.sort()

confmat = torchmetrics.ConfusionMatrix(num_classes=NUM_OF_CLS, compute_on_step=False)
for ind in tqdm(range(len(filelist))):
    label_file = filelist[ind]
    label, seg_logit, pred = com.load_meta(label_file, 'label', 'seg_logit', 'pred')
    mask = label != 255
    confmat(torch.from_numpy(pred[mask]), torch.from_numpy(label[mask]))
confmat = confmat.compute().numpy() # (21, 21)
#%%
with np.errstate(divide='ignore', invalid='ignore'):
    ious = np.diag(confmat) / (confmat.sum(1) + confmat.sum(0) - np.diag(confmat))
accs = confmat.diagonal() / confmat.sum(1)        # (21,)
miou = np.nanmean(ious)
macc = np.nanmean(accs)

#%%
ds_iou = pd.Series(ious, index=np.arange(NUM_OF_CLS))
ds_iou['avg'] = miou
ds_acc = pd.Series(accs, index=np.arange(NUM_OF_CLS))
ds_acc['avg'] = macc

df = pd.DataFrame({'iou': ds_iou, 'acc': ds_acc})
df.to_csv(join(eval_dir, 'per_class_acc_iou.csv'), float_format='%.3f')
