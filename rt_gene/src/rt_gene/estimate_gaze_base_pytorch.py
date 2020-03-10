# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from rt_gene.RTGENEModels import RTGENEModelResnet18
from rt_gene.gaze_tools import get_endpoint


class GazeEstimator(object):
    """This class encapsulates a deep neural network for gaze estimation.

    It retrieves two image streams, one containing the left eye and another containing the right eye.
    It synchronizes these two images with the estimated head pose.
    The images are then converted in a suitable format, and a forward pass of the deep neural network
    results in the estimated gaze for this frame. The estimated gaze is then published in the (theta, phi) notation."""

    def __init__(self, device_id_gaze, model_files):
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.device_id_gazeestimation = device_id_gaze
        self.models = []
        for ckpt in model_files:
            _model = RTGENEModelResnet18(num_out=2)
            _torch_load = torch.load(ckpt)['state_dict']
            _state_dict = {k[7:]: v for k, v in _torch_load.items()}
            _model.load_state_dict(_state_dict)
            _model.to(self.device_id_gazeestimation)
            _model.eval()
            self.models.append(_model)

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        _transformed_left = torch.stack(inference_input_left_list).to(self.device_id_gazeestimation)
        _transformed_right = torch.stack(inference_input_right_list).to(self.device_id_gazeestimation)
        _tranformed_head = torch.as_tensor(inference_headpose_list).to(self.device_id_gazeestimation)

        result = np.array([model(_transformed_left, _transformed_right, _tranformed_head).detach().cpu().squeeze(0).numpy() for model in self.models])
        return result

    @staticmethod
    def visualize_eye_result(eye_image, est_gaze):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(eye_image)

        center_x = output_image.shape[1] / 2
        center_y = output_image.shape[0] / 2

        endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50)

        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))
        return output_image

    def input_from_image(self, cv_image):
        return self._transform(cv_image)
