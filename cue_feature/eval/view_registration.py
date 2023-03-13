#%%
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os.path import exists, join, dirname
import os
import matin

reg_dir = '/LOCAL2/ramdrop/github/point_registration/DPR/logs_kitti/BTL_0513_082202/eval_test_0513_142428/registration'

#%%
print('on successful pairs:\n-----------------')
df = pd.read_csv(join(reg_dir, 'RTE_RRE_RANSAC_FEAT_0_5_ac_100.csv'))
success_baseline = np.logical_and(df['baseline_rte'].to_numpy().astype(np.float32) < 2, df['baseline_rre'].to_numpy().astype(np.float32) < 5)
success_bins = np.logical_and(df['bins_rte'].to_numpy().astype(np.float32) < 2, df['bins_rre'].to_numpy().astype(np.float32) < 5)
print(f"Base: RTE:{df[success_baseline]['baseline_rte'].mean():.3f}, RRE:{df[success_baseline]['baseline_rre'].mean():.3f}, SR:{success_baseline.sum()}/{df['baseline_rte'].to_numpy().shape[0]}")
print(f"Bins: RTE:{df[success_baseline]['bins_rte'].mean():.3f}, RRE:{df[success_baseline]['bins_rre'].mean():.3f}, SR:{success_baseline.sum()}/{df['baseline_rte'].to_numpy().shape[0]}")
print(f"Bins: RTE:{df[success_bins]['bins_rte'].mean():.3f}, RRE:{df[success_bins]['bins_rre'].mean():.3f}, SR:{success_bins.sum()}/{df['baseline_rte'].to_numpy().shape[0]}")
