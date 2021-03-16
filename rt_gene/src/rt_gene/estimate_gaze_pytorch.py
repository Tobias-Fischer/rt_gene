# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os

import cv2
import torch
from torchvision import transforms
from tqdm import tqdm

from rt_gene.estimate_gaze_base import GazeEstimatorBase
from rt_gene.gaze_estimation_models_pytorch import GazeEstimationModelVGG, GazeEstimationModelResnet18
from rt_gene.download_tools import download_gaze_pytorch_models, md5


class GazeEstimator(GazeEstimatorBase):
    def __init__(self, device_id_gaze, model_files, known_hashes=(
            "ae435739673411940eed18c98c29bfb1", "4afd7ccf5619552ed4a9f14606b7f4dd", "743902e643322c40bd78ca36aacc5b4d",
            "06a10f43088651053a65f9b0cd5ac4aa")):
        super(GazeEstimator, self).__init__(device_id_gaze, model_files)
        download_gaze_pytorch_models()
        # check md5 hashes
        _model_hashes = [md5(model) for model in model_files]
        _correct = [1 for hash in _model_hashes if hash not in known_hashes]
        if sum(_correct) > 0:
            raise ImportError(
                "MD5 Hashes of supplied model_files do not match the known_hashes argument. You have probably not set "
                "the --models argument and therefore you are trying to use TensorFlow models. If you are training your "
                "own models, then please supply the md5sum hashes in the known_hashes argument. If you're not, "
                "then you're using old models. The newer models should have downloaded already so please update the "
                "estimate_gaze.launch file that you've modified.")

        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(60, 36), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self._models = []
        for ckpt in self.model_files:
            try:
                _model = GazeEstimationModelVGG(num_out=2)
                _model.load_state_dict(torch.load(ckpt))
                _model.to(self.device_id_gazeestimation)
                _model.eval()
                self._models.append(_model)
            except Exception as e:
                print("Error loading checkpoint", ckpt)
                raise e

        tqdm.write('Loaded ' + str(len(self._models)) + ' model(s)')

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        transformed_left = torch.stack(inference_input_left_list).to(self.device_id_gazeestimation)
        transformed_right = torch.stack(inference_input_right_list).to(self.device_id_gazeestimation)
        tranformed_head = torch.as_tensor(inference_headpose_list).to(self.device_id_gazeestimation)

        result = [model(transformed_left, transformed_right, tranformed_head).detach().cpu() for model in self._models]
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1).numpy()
        result[:, 1] += self._gaze_offset
        return result

    def input_from_image(self, cv_image):
        return self._transform(cv_image)
