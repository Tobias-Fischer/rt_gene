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

torch.backends.cudnn.benchmark = True

root_dir = os.path.dirname(os.path.realpath(__file__))

root_parser = ArgumentParser(add_help=False)
root_parser.add_argument('--hdf5_file', type=str, default=os.path.abspath(os.path.join(root_dir, "../../RT_GENE/dataset.hdf5")))
root_parser.add_argument('--fold_folder', type=str, default=os.path.abspath(os.path.join(root_dir, '../../rt_gene/model_nets/pytorch_checkpoints')))
root_parser.add_argument('--num_io_workers', default=8, type=int)
root_parser.add_argument('--loss_fn', choices=["mse", "pinball"], default="mse")
root_parser.add_argument('--model_base', choices=["vgg", "resnet18", "preactresnet"], default="vgg")
root_parser.add_argument('--batch_size', default=64, type=int)

hyperparams = root_parser.parse_args()

_param_num = {
    "mse": 2,
    "pinball": 3
}
_models = {
    "vgg": partial(GazeEstimationModelVGG, num_out=_param_num.get(hyperparams.loss_fn)),
    "resnet18": partial(GazeEstimationModelResnet18, num_out=_param_num.get(hyperparams.loss_fn)),
    "preactresnet": partial(GazeEstimationModelPreactResnet, num_out=_param_num.get(hyperparams.loss_fn))
}

test_subjects = [[5, 6, 11, 12, 13], [3, 4, 7, 9], [1, 2, 8, 10]]
folds = [os.path.abspath(os.path.join(hyperparams.fold_folder, "fold_{}/".format(i))) for i in range(3)]

criterion = GazeAngleAccuracy()

for fold_idx, (test_subject, fold) in enumerate(zip(test_subjects, folds)):
    # get each checkpoint and see which one is best
    epoch_ckpt = glob.glob(os.path.abspath(os.path.join(fold, "*.ckpt")))
    for ckpt_idx, ckpt in enumerate(epoch_ckpt):
        # load data
        data_test = RTGENEH5Dataset(h5_file=h5py.File(hyperparams.hdf5_file, mode="r"), subject_list=test_subject)
        data_loader = DataLoader(data_test, batch_size=hyperparams.batch_size, shuffle=True, num_workers=hyperparams.num_io_workers, pin_memory=False)

        model = _models.get(hyperparams.model_base)()
        # the ckpt file saves the pytorch_lightning module which includes it's child members. The only child member we're interested in is the "_model".
        # Loading the state_dict with _model creates an error as the GazeEstimationmodelResetnet18 tries to find a child called _model within it that doesn't
        # exist. Thus remove _model from the dictionary and all is well.
        model_state = {k[7:]: v for k, v in torch.load(ckpt)['state_dict'].items()}
        model.load_state_dict(model_state)
        model.to("cuda:0")
        model.eval()

        angle_criterion_acc = []
        p_bar = tqdm(data_loader)
        for left, right, headpose, gaze_labels in p_bar:
            p_bar.set_description("Testing Fold {}, Checkpoint {}...".format(fold_idx, ckpt_idx))
            left = left.to("cuda:0")
            right = right.to("cuda:0")
            headpose = headpose.to("cuda:0")
            angle_out = model(left, right, headpose)
            angle_acc = criterion(angle_out[:, :2], gaze_labels)
            angle_criterion_acc.append(angle_acc)
            del left
            del right
            del headpose

        angle_criterion_acc_arr = np.array(angle_criterion_acc)
        tqdm.write("Fold: {}, Checkpoint: {}, Mean: {}, STD: {}".format(fold_idx, ckpt_idx, np.mean(angle_criterion_acc_arr), np.std(angle_criterion_acc_arr)))

        del model
