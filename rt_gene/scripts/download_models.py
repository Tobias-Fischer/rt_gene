#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
# (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import hashlib
import os

import requests
from tqdm import tqdm


def md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def request_if_not_exist(file_name, url, md5sum=None, chunksize=1024):
    if not os.path.isfile(file_name):
        request = requests.get(url, timeout=10, stream=True)
        with open(file_name, 'wb') as fh:
            # Walk through the request response in chunks of 1MiB
            for chunk in tqdm(request.iter_content(chunksize), desc=os.path.basename(file_name),
                              total=int(int(request.headers['Content-length']) / chunksize),
                              unit="KiB"):
                fh.write(chunk)
        if md5sum is not None:
            print("Checking md5 for {}".format(os.path.basename(file_name)))
            assert md5sum == md5(
                file_name), "MD5Sums do not match for {}. Please) delete the same file name to re-download".format(
                file_name)


def download_gaze_tensorflow_models():
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/Model_allsubjects1.h5'),
        'https://imperialcollegelondon.box.com/shared/static/zu424pzptmw1klh70jsc697b37h7mwif.h5',
        'e55ea59d494d66dd075bf1503a32f99c')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/all_subjects_mpii_prl_utmv_0_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/5cjnijpo8qxawbkik0gjrmyc802j2h1v.h5",
        "af5554d5b405e5a1515c08d553a96613")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/all_subjects_mpii_prl_utmv_1_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/1ye5jlh5ce11f93yn1s36uysjta7a3ob.h5",
        "eccea117ed40c903d07537125f77af88")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/all_subjects_mpii_prl_utmv_2_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/5vl9samndju9zhygtai8z6kkpw2jmjll.h5",
        "e41362688850bd3d51f58c14c75f3744")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/all_subjects_mpii_prl_utmv_3_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/hmcoxopu4xetic5bm47xqrl5mqktpg92.h5",
        "581f7a96ef88faf3a564aca083496dfa")


def download_gaze_pytorch_models():
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/Model_allsubjects1_pytorch.model'),
        "https://imperialcollegelondon.box.com/shared/static/zblg37jitf9q245k3ytad8nv814nz9o8.model",
        "ca13a350902899dd06febb897b111aeb")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/Model_allsubjects2_pytorch.model'),
        "https://imperialcollegelondon.box.com/shared/static/nhmwcwzf2j15x44i4bosi8muurqj0kz7.model",
        "0ee3ec584b6e2ba0a7c2187d78a15a20")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/Model_allsubjects3_pytorch.model'),
        "https://imperialcollegelondon.box.com/shared/static/25anki14qn189ah4lh5gfrhh292utm7p.model",
        "b02c6252a39dcef36edd158aca135f9e")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/Model_allsubjects4_pytorch.model'),
        "https://imperialcollegelondon.box.com/shared/static/5j6mum8350tsn51tcktus1546kwyu6yy.model",
        "e9d2aff52aff1270fcdd8466f97b3528")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../model_nets/Model_prl_mpii_allsubjects1_pytorch.model'),
        "https://imperialcollegelondon.box.com/shared/static/1rp9nrdw7y2q5pbuly4pipf2riy0ne8w.model",
        "0ffac7333ba5659d2b7f86ac302ba9ba")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../model_nets/Model_prl_mpii_allsubjects2_pytorch.model'),
        "https://imperialcollegelondon.box.com/shared/static/nlkqe42rch2so4k5fb9ql40jjxsze0bh.model",
        "87e13756863393a94a8ffc4a75f2fca2")


def download_blink_models():
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/blink_model_1.h5'),
                         "https://imperialcollegelondon.box.com/shared/static/lke3k5f86qnfchzfh6lpon3isniqvkpz.h5",
                         "75aab57645faed3beaba5dedfd0f3d36")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/blink_model_2.h5'),
                         "https://imperialcollegelondon.box.com/shared/static/x4u8c5mr468r6wzki93v45jemf3sz0r5.h5",
                         "ed994ea8384a7894dac04926601d06ff")


def download_external_landmark_models():
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/ThreeDDFA/w_shp_sim.npy'),
        'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_shp_sim.npy?raw=true',
        '74d41d465580924456defd99401cfcdd')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/ThreeDDFA/w_exp_sim.npy'),
        'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_exp_sim.npy?raw=true',
        '7566556b7760b8691f3a6fddbdd38fcc')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/phase1_wpdc_vdc.pth.tar'),
        'https://github.com/cleardusk/3DDFA/blob/master/models/phase1_wpdc_vdc.pth.tar?raw=true',
        '01054c039b12b1b5f6e34e1fcf44fbf6')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../model_nets/dlib_face_recognition_resnet_model_v1.dat'),
        'https://imperialcollegelondon.box.com/shared/static/7zfltrwhrss0zsq2d2z0mgbz4j3ij3fk.dat',
        '2316b25ae80acf4ad9b620b00071c423')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../model_nets/SFD/s3fd_facedetector.pth'),
        "https://imperialcollegelondon.box.com/shared/static/wgfkq3pyzzuewiiwq0pzj0xiebolvlju.pth",
        "3b5a9888bf0beb93c177db5a18375a6c")


if __name__ == '__main__':
    download_gaze_tensorflow_models()
    download_gaze_pytorch_models()
    download_blink_models()
    download_external_landmark_models()
