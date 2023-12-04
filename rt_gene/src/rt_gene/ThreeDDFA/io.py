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


import os
import numpy as np
import torch
import pickle
import scipy.io as sio


def mkdir(d):
    """only works on *nix system"""
    if not os.path.isdir(d) and not os.path.exists(d):
        os.system('mkdir -p {}'.format(d))


def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]


def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))


def _dump(wfp, obj):
    suffix = _get_suffix(wfp)
    if suffix == 'npy':
        np.save(wfp, obj)
    elif suffix == 'pkl':
        pickle.dump(obj, open(wfp, 'wb'))
    else:
        raise Exception('Unknown Type: {}'.format(suffix))


def _load_tensor(fp, mode='cpu'):
    if mode.lower() == 'cpu':
        return torch.from_numpy(_load(fp))
    elif mode.lower() == 'gpu':
        return torch.from_numpy(_load(fp)).cuda()


def _tensor_to_cuda(x):
    if x.is_cuda:
        return x
    else:
        return x.cuda()


def _load_gpu(fp):
    return torch.from_numpy(_load(fp)).cuda()


def load_bfm(model_path):
    suffix = _get_suffix(model_path)
    if suffix == 'mat':
        C = sio.loadmat(model_path)
        model = C['model_refine']
        model = model[0, 0]

        model_new = {}
        w_shp = model['w'].astype(float)
        model_new['w_shp_sim'] = w_shp[:, :40]
        w_exp = model['w_exp'].astype(float)
        model_new['w_exp_sim'] = w_exp[:, :10]

        u_shp = model['mu_shape']
        u_exp = model['mu_exp']
        u = (u_shp + u_exp).astype(float)
        model_new['mu'] = u
        model_new['tri'] = model['tri'].astype(int) - 1

        # flatten it, pay attention to index value
        keypoints = model['keypoints'].astype(int) - 1
        keypoints = np.concatenate((3 * keypoints, 3 * keypoints + 1, 3 * keypoints + 2), axis=0)

        model_new['keypoints'] = keypoints.T.flatten()

        #
        w = np.concatenate((w_shp, w_exp), axis=1)
        w_base = w[keypoints]
        w_norm = np.linalg.norm(w, axis=0)
        w_base_norm = np.linalg.norm(w_base, axis=0)

        dim = w_shp.shape[0] // 3
        u_base = u[keypoints].reshape(-1, 1)
        w_shp_base = w_shp[keypoints]
        w_exp_base = w_exp[keypoints]

        model_new['w_norm'] = w_norm
        model_new['w_base_norm'] = w_base_norm
        model_new['dim'] = dim
        model_new['u_base'] = u_base
        model_new['w_shp_base'] = w_shp_base
        model_new['w_exp_base'] = w_exp_base

        _dump(model_path.replace('.mat', '.pkl'), model_new)
        return model_new
    else:
        return _load(model_path)


_load_cpu = _load
_numpy_to_tensor = lambda x: torch.from_numpy(x)
_tensor_to_numpy = lambda x: x.cpu()
_numpy_to_cuda = lambda x: _tensor_to_cuda(torch.from_numpy(x))
_cuda_to_tensor = lambda x: x.cpu()
_cuda_to_numpy = lambda x: x.cpu().numpy()
