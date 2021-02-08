import os
from argparse import ArgumentParser
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_curve, precision_recall_curve
from torch import sigmoid
from tqdm import tqdm

from rt_bene.blink_estimation_models_pytorch import BlinkEstimationModelResnet18
from rt_bene_model_training.pytorch.rtbene_dataset import RTBENEH5Dataset

if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.realpath(__file__))
    _root_parser = ArgumentParser(add_help=False)
    _root_parser.add_argument('--hdf5_file', type=str,
                              default=os.path.abspath(os.path.join(root_dir, "../../RT_BENE/rtbene_dataset.hdf5")))
    _root_parser.add_argument('--dataset', type=str, choices=["rt_bene"], default="rt_bene")
    _root_parser.add_argument('--model_net_dir', type=str, required=True)
    _args = _root_parser.parse_args()

    _valid_subjects = [0, 11, 15, 16]
    # create a master list of predictions and labels
    fig, (roc_fig, prc_fig) = plt.subplots(1, 2)
    for model_file in tqdm(glob.glob(os.path.join(_args.model_net_dir, "*.model")), desc="Model", position=1):
        labels = []
        predictions = []
        _model = BlinkEstimationModelResnet18()
        _model.load_state_dict(torch.load(model_file))
        _model.cuda()
        _model.eval()
        for subject in _valid_subjects:
            _data_validate = RTBENEH5Dataset(h5_file=h5py.File(_args.hdf5_file, mode="r"),
                                             subject_list=[subject], loader_desc=subject)
            for index in tqdm(range(0, len(_data_validate)), desc=f"Inferring for subject {subject}", position=2):
                _left, _right, _label = _data_validate[index]
                _predicted_blink = sigmoid(
                    _model(_left.unsqueeze(0).float().cuda(), _right.unsqueeze(0).float().cuda()))

                labels.extend(_label)
                predictions.append(float(_predicted_blink.detach().cpu()))

        fpr, tpr, ft_thresholds = roc_curve(y_true=labels, y_score=predictions)
        positive_predictive_value, sensitivity, pr_thresholds = precision_recall_curve(labels, predictions)

        optimal_roc_idx = np.argmax(tpr - fpr)
        optimal_roc_threshold = ft_thresholds[optimal_roc_idx]
        optimal_prc_idx = np.argmax(sensitivity + positive_predictive_value)
        optimal_prc_threshold = pr_thresholds[optimal_prc_idx]
        print(f"Optimal Threshold from ROC {optimal_roc_threshold}, optimal threshold from PRC {optimal_prc_threshold}")

        prc_fig.step(sensitivity, positive_predictive_value, where='post')
        prc_fig.set_xlabel('Sensitivity (Recall)')
        prc_fig.set_ylabel('Positive Predictive Value (Precision)')
        prc_fig.set_ylim([0.0, 1.05])
        prc_fig.set_xlim([0.0, 1.05])
        prc_fig.set_title("Precision recall curve")

        roc_fig.plot(fpr, tpr, lw=2)
        roc_fig.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        roc_fig.set_xlim([0.0, 1.05])
        roc_fig.set_ylim([0.0, 1.05])
        roc_fig.set_xlabel('1 - Sensitivity')
        roc_fig.set_ylabel('Specificity')
        roc_fig.set_title('Receiver operating characteristics curve')

    plt.show()
