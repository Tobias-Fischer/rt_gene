# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from rt_gene.estimate_gaze_base import GazeEstimatorBase
from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelResnet18


class GazeEstimator(GazeEstimatorBase):
    """This class encapsulates a deep neural network for gaze estimation.

    It retrieves two image streams, one containing the left eye and another containing the right eye.
    It synchronizes these two images with the estimated head pose.
    The images are then converted in a suitable format, and a forward pass of the deep neural network
    results in the estimated gaze for this frame. The estimated gaze is then published in the (theta, phi) notation."""

    def __init__(self, device_id_gaze, model_files):
        super(GazeEstimator, self).__init__(device_id_gaze, model_files)
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self._models = []
        for ckpt in self.model_files:
            _model = GazeEstimationModelResnet18(num_out=2)
            _torch_load = torch.load(ckpt)['state_dict']

            # the ckpt file saves the pytorch_lightning module which includes it's child members. The only child member we're interested in is the "_model".
            # Loading the state_dict with _model creates an error as the GazeEstimationmodelResetnet18 tries to find a child called _model within it that doesn't
            # exist. Thus remove _model from the dictionary and all is well.
            _state_dict = {k[7:]: v for k, v in _torch_load.items()}
            _model.load_state_dict(_state_dict)
            _model.to(self.device_id_gazeestimation)
            _model.eval()
            self._models.append(_model)
        tqdm.write('Loaded ' + str(len(self._models)) + ' model(s)')

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        transformed_left = torch.stack(inference_input_left_list).to(self.device_id_gazeestimation)
        transformed_right = torch.stack(inference_input_right_list).to(self.device_id_gazeestimation)
        tranformed_head = torch.as_tensor(inference_headpose_list).to(self.device_id_gazeestimation)

        result = [model(transformed_left, transformed_right, tranformed_head).detach().cpu() for model in self._models]
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1).numpy()
        return result

    def input_from_image(self, cv_image):
        return self._transform(cv_image)
