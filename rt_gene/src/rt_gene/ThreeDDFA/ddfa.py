#!/usr/bin/env python3
# coding: utf-8

import numpy as np

import torch
from .params import *


def _parse_param(param):
    """Work for both numpy and tensor"""
    p_ = param[:12].reshape(3, -1)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(-1, 1)
    alpha_exp = param[52:].reshape(-1, 1)
    return p, offset, alpha_shp, alpha_exp


def reconstruct_vertex(param, whitening=True, dense=False, transform=True):
    """Whitening param -> 3d vertex, based on the 3dmm param: u_base, w_shp, w_exp
    dense: if True, return dense vertex, else return 68 sparse landmarks. All dense or sparse vertex is transformed to
    image coordinate space, but without alignment caused by face cropping.
    transform: whether transform to image space
    """
    if len(param) == 12:
        param = np.concatenate((param, [0] * 50))
    if whitening:
        if len(param) == 62:
            param = param * param_std + param_mean
        else:
            param = np.concatenate((param[:11], [0], param[11:]))
            param = param * param_std + param_mean

    p, offset, alpha_shp, alpha_exp = _parse_param(param)

    if dense:
        t1 = np.dot(w_shp, alpha_shp)
        t2 = np.dot(w_exp, alpha_exp)
        vertex = np.matmul(p, (u + t1 + t2).reshape(3, -1, order='F')) + offset
    else:
        """For 68 pts"""
        t1 = np.dot(w_shp_base, alpha_shp)
        t2 = np.dot(w_exp_base, alpha_exp)
        vertex = np.matmul(p, (u_base + t1 + t2).reshape(3, -1, order='F')) + offset

    if transform:
        # transform to image coordinate space
        vertex[1, :] = std_size + 1 - vertex[1, :]

    return vertex


class ToTensorGjz(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'


class NormalizeGjz(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor
