#%%
import os
from os.path import join
from glob import glob
import shutil
import wandb
api = wandb.Api()
runs = api.runs('ramdrop/FastPointTransformer-SCANNET')

log_dir = 'logs_scannet'

runs_remote = []
for run in runs:
    # print(run.name)
    runs_remote.append(run.name)

runs_local = glob(join(log_dir, '*'))
for run in runs_local:
    if run.split('/')[-1] not in runs_remote:
        print(run)

confirm = input("Confirm you want to delete them? Y/[N]:")
if confirm == 'y':   
    for run in runs_local:
        if run.split('/')[-1] not in runs_remote:
            shutil.rmtree(run)
    print('deleted')
else:
    print('aborted.')