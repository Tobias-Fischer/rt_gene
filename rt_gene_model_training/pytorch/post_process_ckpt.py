import os
from argparse import ArgumentParser
from functools import partial
from glob import glob
from pathlib import Path

import torch
from tqdm import tqdm

from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18, GazeEstimationModelVGG, GazeEstimationModelPreactResnet

if __name__ == "__main__":
    _root_parser = ArgumentParser(add_help=False)
    root_dir = os.path.dirname(os.path.realpath(__file__))
    _root_parser.add_argument('--ckpt_dir', type=str, default=os.path.abspath(os.path.join(root_dir, '../../rt_gene_model_training/pytorch/checkpoints/fold_0/')))
    _root_parser.add_argument('--save_dir', type=str, default=os.path.abspath(os.path.join(root_dir, '../../rt_gene/model_nets/pytorch_models/')))
    _root_parser.add_argument('--model_base', choices=["vgg16", "resnet18", "preactresnet"], default="vgg16")
    _root_parser.add_argument('--loss_fn', choices=["mse", "pinball"], default="mse")
    _params = _root_parser.parse_args()

    _param_num = {
        "mse": 2,
        "pinball": 3
    }
    _models = {
        "vgg16": partial(GazeEstimationModelVGG, num_out=_param_num.get(_params.loss_fn)),
        "resnet18": partial(GazeEstimationModelResnet18, num_out=_param_num.get(_params.loss_fn)),
        "preactresnet": partial(GazeEstimationModelPreactResnet, num_out=_param_num.get(_params.loss_fn))
    }

    # create save dir
    Path(_params.save_dir).mkdir(parents=True, exist_ok=True)

    _model = _models.get(_params.model_base)()
    for ckpt in tqdm(glob(os.path.join(_params.ckpt_dir, "*.ckpt"))):
        filename, file_extension = os.path.splitext(ckpt)
        filename = os.path.basename(filename)
        _torch_load = torch.load(ckpt)['state_dict']

        # the ckpt file saves the pytorch_lightning module which includes it's child members. The only child member we're interested in is the "_model".
        # Loading the state_dict with _model creates an error as the model tries to find a child called _model within it that doesn't
        # exist. Thus remove _model from the dictionary and all is well.
        _model_prefix = "_model."
        _state_dict = {k[len(_model_prefix):]: v for k, v in _torch_load.items() if k.startswith(_model_prefix)}
        _model.load_state_dict(_state_dict)
        torch.save(_model.state_dict(), os.path.join(_params.save_dir, "{}.model").format(filename))
