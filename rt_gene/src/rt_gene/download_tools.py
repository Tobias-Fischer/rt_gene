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
        print("Download model: {}".format(os.path.basename(file_name)))
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
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/Model_allsubjects1.h5'),
        'https://imperialcollegelondon.box.com/shared/static/zu424pzptmw1klh70jsc697b37h7mwif.h5',
        'e55ea59d494d66dd075bf1503a32f99c')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/all_subjects_mpii_prl_utmv_0_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/5cjnijpo8qxawbkik0gjrmyc802j2h1v.h5",
        "af5554d5b405e5a1515c08d553a96613")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/all_subjects_mpii_prl_utmv_1_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/1ye5jlh5ce11f93yn1s36uysjta7a3ob.h5",
        "eccea117ed40c903d07537125f77af88")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/all_subjects_mpii_prl_utmv_2_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/5vl9samndju9zhygtai8z6kkpw2jmjll.h5",
        "e41362688850bd3d51f58c14c75f3744")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/all_subjects_mpii_prl_utmv_3_02.h5'),
        "https://imperialcollegelondon.box.com/shared/static/hmcoxopu4xetic5bm47xqrl5mqktpg92.h5",
        "581f7a96ef88faf3a564aca083496dfa")


def download_gaze_pytorch_models():
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model'),
        "https://imperialcollegelondon.box.com/shared/static/6rvctw7wmkpl7a9bw9hm1b9b7dwntfut.model",
        "ae435739673411940eed18c98c29bfb1")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects2.model'),
        "https://imperialcollegelondon.box.com/shared/static/xuhs5qg7eju4kw3e4to7db945qk2c123.model",
        "4afd7ccf5619552ed4a9f14606b7f4dd")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects3.model'),
        "https://imperialcollegelondon.box.com/shared/static/h75tro719fcyvgdkzr8tarpco32ve21u.model",
        "743902e643322c40bd78ca36aacc5b4d")
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/gaze_model_pytorch_vgg16_prl_mpii_allsubjects4.model'),
        "https://imperialcollegelondon.box.com/shared/static/1xywt1so20vw09iij4t3tp9lu6f6yb0g.model",
        "06a10f43088651053a65f9b0cd5ac4aa")


def download_blink_tensorflow_models():
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/blink_model_1.h5'),
                         "https://imperialcollegelondon.box.com/shared/static/lke3k5f86qnfchzfh6lpon3isniqvkpz.h5",
                         "75aab57645faed3beaba5dedfd0f3d36")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/blink_model_2.h5'),
                         "https://imperialcollegelondon.box.com/shared/static/x4u8c5mr468r6wzki93v45jemf3sz0r5.h5",
                         "ed994ea8384a7894dac04926601d06ff")


def download_blink_pytorch_models():
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/blink_model_pytorch_vgg16_allsubjects1.model'),
                         "https://imperialcollegelondon.box.com/shared/static/wwky1um443vgz9oy90zllv0s7474a5dj.model",
                         "cde99055e3b6dcf9fae6b78191c0fd9b")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../model_nets/blink_model_pytorch_vgg16_allsubjects2.model'),
                         "https://imperialcollegelondon.box.com/shared/static/psha8bclz9bv5d87qetajgovioc03vb3.model",
                         "67339ceefcfec4b3b8b3d7ccb03fadfa")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../model_nets/blink_model_pytorch_vgg16_allsubjects3.model'),
                         "https://imperialcollegelondon.box.com/shared/static/puebet9v05pxt06g42rtz805u5ww0u7e.model",
                         "e5de548b2a97162c5e655259463e4d23")

    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../model_nets/blink_model_pytorch_resnet18_allsubjects1.model'),
                         "https://imperialcollegelondon.box.com/shared/static/p8jmekxhw4k8xtbz6vph3924g6ywnbre.model",
                         "7c228fe7b95ce5960c4c5cae8f2d3a09")
    request_if_not_exist(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      '../../model_nets/blink_model_pytorch_resnet18_allsubjects2.model'),
                         "https://imperialcollegelondon.box.com/shared/static/yu53g8n1007d80s71o3dyrxqtv59o15u.model",
                         "0a0d2d066737b333737018d738de386f")




def download_external_landmark_models():
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/ThreeDDFA/w_shp_sim.npy'),
        'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_shp_sim.npy?raw=true',
        '74d41d465580924456defd99401cfcdd')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/ThreeDDFA/w_exp_sim.npy'),
        'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_exp_sim.npy?raw=true',
        '7566556b7760b8691f3a6fddbdd38fcc')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/phase1_wpdc_vdc.pth.tar'),
        'https://github.com/cleardusk/3DDFA/blob/master/models/phase1_wpdc_vdc.pth.tar?raw=true',
        '01054c039b12b1b5f6e34e1fcf44fbf6')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     '../../model_nets/dlib_face_recognition_resnet_model_v1.dat'),
        'https://imperialcollegelondon.box.com/shared/static/7zfltrwhrss0zsq2d2z0mgbz4j3ij3fk.dat',
        '2316b25ae80acf4ad9b620b00071c423')
    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/SFD/s3fd_facedetector.pth'),
        "https://imperialcollegelondon.box.com/shared/static/wgfkq3pyzzuewiiwq0pzj0xiebolvlju.pth",
        "3b5a9888bf0beb93c177db5a18375a6c")

    request_if_not_exist(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../model_nets/dlib_face_recognition_resnet_model_v1.dat'),
        "https://imperialcollegelondon.box.com/shared/static/mmo544mmrwei3tqc8jfe5fim104g57z9.dat",
        "2316b25ae80acf4ad9b620b00071c423")
