#! /usr/bin/env python

import os
from argparse import ArgumentParser
from functools import partial
from glob import glob
from pathlib import Path
import torch
from tqdm import tqdm
from rt_bene.blink_estimation_models_pytorch import BlinkEstimationModelResnet18, BlinkEstimationModelResnet50, \
    BlinkEstimationModelVGG16, BlinkEstimationModelVGG19, BlinkEstimationModelDenseNet121

if __name__ == "__main__":
    _root_parser = ArgumentParser(add_help=False)
    root_dir = os.path.dirname(os.path.realpath(__file__))
    _root_parser.add_argument('--ckpt_dir', type=str, default=os.path.abspath(
        os.path.join(root_dir, '../../rt_bene_model_training/pytorch/checkpoints/')))
    _root_parser.add_argument('--save_dir', type=str, default=os.path.abspath(
        os.path.join(root_dir, '../../rt_bene_model_training/pytorch/model_nets/')))
    _root_parser.add_argument('--model_base', choices=["vgg16", "vgg19", "resnet18", "resnet50", "densenet121"],
                              default="densenet121")
    _params = _root_parser.parse_args()

    _models = {
        "resnet18": BlinkEstimationModelResnet18,
        "resnet50": BlinkEstimationModelResnet50,
        "vgg16": BlinkEstimationModelVGG16,
        "vgg19": BlinkEstimationModelVGG19,
        "densenet121": BlinkEstimationModelDenseNet121
    }

    # create save dir
    Path(_params.save_dir).mkdir(parents=True, exist_ok=True)

    _model = _models.get(_params.model_base)()
    for ckpt in tqdm(glob(os.path.join(_params.ckpt_dir, "*.ckpt")), desc="Processing..."):
        filename, file_extension = os.path.splitext(ckpt)
        filename = os.path.basename(filename)
        _torch_load = torch.load(ckpt)['state_dict']

        # the ckpt file saves the pytorch_lightning module which includes it's child members. The only child member we're interested in is the "_model".
        # Loading the state_dict with _model creates an error as the model tries to find a child called _model within it that doesn't
        # exist. Thus remove _model from the dictionary and all is well.
        _state_dict = dict(_torch_load.items())
        _state_dict = {k[7:]: v for k, v in _state_dict.items() if k.startswith("_model.")}
        _model.load_state_dict(_state_dict)
        torch.save(_model.state_dict(), os.path.join(_params.save_dir, f"{filename}.model"))
