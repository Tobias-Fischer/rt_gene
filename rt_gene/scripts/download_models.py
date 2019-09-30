#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

import os
import os.path as osp
from six.moves import urllib
import hashlib


def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download_if_not_exist(fname, url, md5sum=None):
    print('Download ' + os.path.basename(fname))
    if not osp.isfile(fname):
        urllib.request.urlretrieve(url, fname)
    if md5sum is not None:
        print('File exists, checking md5')
        assert md5sum == md5(fname)


if __name__ == '__main__':
    download_if_not_exist(osp.join(osp.dirname(osp.realpath(__file__)), '../model_nets/ThreeDDFA/w_shp_sim.npy'),
                          'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_shp_sim.npy?raw=true',
                          '74d41d465580924456defd99401cfcdd')
    download_if_not_exist(osp.join(osp.dirname(osp.realpath(__file__)), '../model_nets/ThreeDDFA/w_exp_sim.npy'),
                          'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_exp_sim.npy?raw=true',
                          '7566556b7760b8691f3a6fddbdd38fcc')
    download_if_not_exist(osp.join(osp.dirname(osp.realpath(__file__)), '../model_nets/phase1_wpdc_vdc.pth.tar'),
                          'https://github.com/cleardusk/3DDFA/blob/master/models/phase1_wpdc_vdc.pth.tar?raw=true',
                          '01054c039b12b1b5f6e34e1fcf44fbf6')
    download_if_not_exist(osp.join(osp.dirname(osp.realpath(__file__)), '../model_nets/Model_allsubjects1.h5'),
                          'https://imperialcollegelondon.box.com/shared/static/zu424pzptmw1klh70jsc697b37h7mwif.h5',
                          'e55ea59d494d66dd075bf1503a32f99c')
    download_if_not_exist(osp.join(osp.dirname(osp.realpath(__file__)), '../model_nets/dlib_face_recognition_resnet_model_v1.dat'),
                          'https://imperialcollegelondon.box.com/shared/static/7zfltrwhrss0zsq2d2z0mgbz4j3ij3fk.dat',
                          '2316b25ae80acf4ad9b620b00071c423')
