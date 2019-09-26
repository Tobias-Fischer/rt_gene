#!/usr/bin/env python3
# coding: utf-8

import sys
import os.path as osp
from .io import _load
import rospkg


d = rospkg.RosPack().get_path('rt_gene') + '/model_nets/ThreeDDFA/'
keypoints = _load(osp.join(d, 'keypoints_sim.npy'))
w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
w_exp = _load(osp.join(d, 'w_exp_sim.npy'))  # simplified version
if sys.version_info > (3, 0):
    meta = _load(osp.join(d, 'param_whitening.pkl'))
else:
    meta = _load(osp.join(d, 'param_whitening_py2.pkl'))
# # param_mean and param_std are used for re-whitening
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = _load(osp.join(d, 'u_shp.npy'))
u_exp = _load(osp.join(d, 'u_exp.npy'))
u = u_shp + u_exp

# for inference
dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120
