import glob
import os
from argparse import ArgumentParser
from functools import partial

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18, \
    GazeEstimationModelVGG, GazeEstimationModelPreactResnet
from rtgene_dataset import RTGENEH5Dataset
from utils.GazeAngleAccuracy import GazeAngleAccuracy


def test_fold(d_loader, model_list, fold_idx, model_idx="Ensemble"):
    assert type(model_list) is list, "model_list should be a list of models"
    angle_criterion_acc = []
    p_bar = tqdm(d_loader)
    for left, right, headpose, gaze_labels in p_bar:
        p_bar.set_description("Testing Fold {}, Model \"{}\"...".format(fold_idx, model_idx))
        left = left.to("cuda:0")
        right = right.to("cuda:0")
        headpose = headpose.to("cuda:0")
        angle_out = [_m(left, right, headpose).detach().cpu() for _m in model_list]
        angle_out = torch.stack(angle_out, dim=1)
        angle_out = torch.mean(angle_out, dim=1)
        angle_acc = criterion(angle_out[:, :2], gaze_labels)
        angle_criterion_acc.append(angle_acc)

    angle_criterion_acc_arr = np.array(angle_criterion_acc)
    tqdm.write(
        "\r\n\tFold: {}, Model: {}, Mean: {}, STD: {}".format(fold_idx, model_idx, np.mean(angle_criterion_acc_arr), np.std(angle_criterion_acc_arr)))


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    root_dir = os.path.dirname(os.path.realpath(__file__))

    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--model_loc', type=str, required=False, help='path to the model files to evaluate', action="append")
    root_parser.add_argument('--hdf5_file', type=str, default=os.path.abspath(os.path.join(root_dir, "../../RT_GENE/rtgene_dataset.hdf5")))
    root_parser.add_argument('--num_io_workers', default=8, type=int)
    root_parser.add_argument('--loss_fn', choices=["mse", "pinball"], default="mse")
    root_parser.add_argument('--model_base', choices=["vgg", "resnet18_0", "preactresnet"], default="vgg")
    root_parser.add_argument('--batch_size', default=64, type=int)

    hyperparams = root_parser.parse_args()

    _param_num = {
        "mse": 2,
        "pinball": 3
    }
    _models = {
        "vgg": partial(GazeEstimationModelVGG, num_out=_param_num.get(hyperparams.loss_fn)),
        "resnet18_0": partial(GazeEstimationModelResnet18, num_out=_param_num.get(hyperparams.loss_fn)),
        "preactresnet": partial(GazeEstimationModelPreactResnet, num_out=_param_num.get(hyperparams.loss_fn))
    }

    test_subjects = [[5, 6, 11, 12, 13], [3, 4, 7, 9], [1, 2, 8, 10]]
    criterion = GazeAngleAccuracy()

    # definition of an ensemble is a list of FILES, if any are folders, then not an ensemble
    ensemble = sum([os.path.isfile(s) for s in hyperparams.model_loc]) == len(hyperparams.model_loc)

    if ensemble:
        _models_list = []
        for model_file in tqdm(hyperparams.model_loc, desc="Ensemble Evaluation; Loading models..."):
            _model = _models.get(hyperparams.model_base)()
            _model.load_state_dict(torch.load(model_file))
            _model.to("cuda:0")
            _model.eval()
            _models_list.append(_model)

        for fold_idx, test_subject in enumerate(test_subjects):
            data_test = RTGENEH5Dataset(h5_file=h5py.File(hyperparams.hdf5_file, mode="r"), subject_list=test_subject)
            data_loader = DataLoader(data_test, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_io_workers, pin_memory=False)
            test_fold(data_loader, fold_idx=fold_idx, model_list=_models_list)
    else:
        folds = [os.path.abspath(os.path.join(hyperparams.model_loc, "fold_{}/".format(i))) for i in range(3)]
        tqdm.write("Every model in fold evaluation (i.e single model)")
        for fold_idx, (test_subject, fold) in enumerate(zip(test_subjects, folds)):
            # get each checkpoint and see which one is best
            epoch_ckpt = glob.glob(os.path.abspath(os.path.join(fold, "*.ckpt")))
            for ckpt in tqdm(epoch_ckpt, desc="Checkpoint evaluation.."):
                # load data
                data_test = RTGENEH5Dataset(h5_file=h5py.File(hyperparams.hdf5_file, mode="r"), subject_list=test_subject)
                data_loader = DataLoader(data_test, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_io_workers, pin_memory=False)

                model = _models.get(hyperparams.model_base)()
                model.load_state_dict(torch.load(ckpt))
                model.to("cuda:0")
                model.eval()

                test_fold(data_loader, model_list=[model], fold_idx=fold_idx, model_idx=os.path.basename(ckpt))
