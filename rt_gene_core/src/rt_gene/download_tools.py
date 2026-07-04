import hashlib
import os
from pathlib import Path

import requests
from tqdm import tqdm

from rt_gene_core.paths import model_path


def md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def request_if_not_exist(file_name, url, md5sum=None, chunksize=1024):
    file_name = Path(file_name)
    if file_name.is_file():
        return

    file_name.parent.mkdir(parents=True, exist_ok=True)
    part_file = file_name.with_name(file_name.name + ".part")
    if part_file.exists():
        part_file.unlink()

    print("Download model: {}".format(file_name.name))
    request = requests.get(url, timeout=10, stream=True)
    try:
        request.raise_for_status()
        content_length = request.headers.get("content-length")
        total = None if content_length is None else max(1, int(content_length) // chunksize)
        with part_file.open("wb") as fh:
            for chunk in tqdm(request.iter_content(chunksize), desc=file_name.name, total=total, unit="KiB"):
                if chunk:
                    fh.write(chunk)
        if md5sum is not None:
            print("Checking md5 for {}".format(file_name.name))
            actual = md5(part_file)
            if md5sum != actual:
                raise RuntimeError(
                    "MD5 checksum mismatch for {}: expected {}, got {}. Delete the file and retry.".format(
                        file_name, md5sum, actual
                    )
                )
        part_file.replace(file_name)
    except Exception:
        if part_file.exists():
            part_file.unlink()
        raise
    finally:
        request.close()


def download_gaze_pytorch_models():
    request_if_not_exist(
        model_path('gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model', writable=True),
        "https://imperialcollegelondon.box.com/shared/static/6rvctw7wmkpl7a9bw9hm1b9b7dwntfut.model",
        "ae435739673411940eed18c98c29bfb1")
    request_if_not_exist(
        model_path('gaze_model_pytorch_vgg16_prl_mpii_allsubjects2.model', writable=True),
        "https://imperialcollegelondon.box.com/shared/static/xuhs5qg7eju4kw3e4to7db945qk2c123.model",
        "4afd7ccf5619552ed4a9f14606b7f4dd")
    request_if_not_exist(
        model_path('gaze_model_pytorch_vgg16_prl_mpii_allsubjects3.model', writable=True),
        "https://imperialcollegelondon.box.com/shared/static/h75tro719fcyvgdkzr8tarpco32ve21u.model",
        "743902e643322c40bd78ca36aacc5b4d")
    request_if_not_exist(
        model_path('gaze_model_pytorch_vgg16_prl_mpii_allsubjects4.model', writable=True),
        "https://imperialcollegelondon.box.com/shared/static/1xywt1so20vw09iij4t3tp9lu6f6yb0g.model",
        "06a10f43088651053a65f9b0cd5ac4aa")


def download_blink_pytorch_models():
    request_if_not_exist(model_path('blink_model_pytorch_vgg16_allsubjects1.model', writable=True),
                         "https://imperialcollegelondon.box.com/shared/static/wwky1um443vgz9oy90zllv0s7474a5dj.model",
                         "cde99055e3b6dcf9fae6b78191c0fd9b")
    request_if_not_exist(model_path('blink_model_pytorch_vgg16_allsubjects2.model', writable=True),
                         "https://imperialcollegelondon.box.com/shared/static/psha8bclz9bv5d87qetajgovioc03vb3.model",
                         "67339ceefcfec4b3b8b3d7ccb03fadfa")
    request_if_not_exist(model_path('blink_model_pytorch_vgg16_allsubjects3.model', writable=True),
                         "https://imperialcollegelondon.box.com/shared/static/puebet9v05pxt06g42rtz805u5ww0u7e.model",
                         "e5de548b2a97162c5e655259463e4d23")

    request_if_not_exist(model_path('blink_model_pytorch_resnet18_allsubjects1.model', writable=True),
                         "https://imperialcollegelondon.box.com/shared/static/p8jmekxhw4k8xtbz6vph3924g6ywnbre.model",
                         "7c228fe7b95ce5960c4c5cae8f2d3a09")
    request_if_not_exist(model_path('blink_model_pytorch_resnet18_allsubjects2.model', writable=True),
                         "https://imperialcollegelondon.box.com/shared/static/yu53g8n1007d80s71o3dyrxqtv59o15u.model",
                         "0a0d2d066737b333737018d738de386f")




def download_external_landmark_models():
    request_if_not_exist(
        model_path('ThreeDDFA/w_shp_sim.npy', writable=True),
        'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_shp_sim.npy?raw=true',
        '74d41d465580924456defd99401cfcdd')
    request_if_not_exist(
        model_path('ThreeDDFA/w_exp_sim.npy', writable=True),
        'https://github.com/cleardusk/3DDFA/blob/master/train.configs/w_exp_sim.npy?raw=true',
        '7566556b7760b8691f3a6fddbdd38fcc')
    request_if_not_exist(
        model_path('phase1_wpdc_vdc.pth.tar', writable=True),
        'https://github.com/cleardusk/3DDFA/blob/master/models/phase1_wpdc_vdc.pth.tar?raw=true',
        '01054c039b12b1b5f6e34e1fcf44fbf6')
    request_if_not_exist(
        model_path('dlib_face_recognition_resnet_model_v1.dat', writable=True),
        'https://imperialcollegelondon.box.com/shared/static/7zfltrwhrss0zsq2d2z0mgbz4j3ij3fk.dat',
        '2316b25ae80acf4ad9b620b00071c423')
    request_if_not_exist(
        model_path('SFD/s3fd_facedetector.pth', writable=True),
        "https://imperialcollegelondon.box.com/shared/static/wgfkq3pyzzuewiiwq0pzj0xiebolvlju.pth",
        "3b5a9888bf0beb93c177db5a18375a6c")

    request_if_not_exist(
        model_path('dlib_face_recognition_resnet_model_v1.dat', writable=True),
        "https://imperialcollegelondon.box.com/shared/static/mmo544mmrwei3tqc8jfe5fim104g57z9.dat",
        "2316b25ae80acf4ad9b620b00071c423")
