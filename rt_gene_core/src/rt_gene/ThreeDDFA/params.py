"""
MIT License

Copyright (c) 2018 Jianzhu Guo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import sys
from rt_gene_core.paths import model_path
from .io import _load


keypoints = _load(model_path('ThreeDDFA/keypoints_sim.npy'))
w_shp = _load(model_path('ThreeDDFA/w_shp_sim.npy'))
w_exp = _load(model_path('ThreeDDFA/w_exp_sim.npy'))  # simplified version
if sys.version_info > (3, 0):
    meta = _load(model_path('ThreeDDFA/param_whitening.pkl'))
else:
    meta = _load(model_path('ThreeDDFA/param_whitening_py2.pkl'))
# # param_mean and param_std are used for re-whitening
param_mean = meta.get('param_mean')
param_std = meta.get('param_std')
u_shp = _load(model_path('ThreeDDFA/u_shp.npy'))
u_exp = _load(model_path('ThreeDDFA/u_exp.npy'))
u = u_shp + u_exp

# for inference
dim = w_shp.shape[0] // 3
u_base = u[keypoints].reshape(-1, 1)
w_shp_base = w_shp[keypoints]
w_exp_base = w_exp[keypoints]
std_size = 120
