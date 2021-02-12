#! /usr/bin/env python

from tqdm import tqdm
from rt_gene.download_tools import download_blink_models
from rt_bene.estimate_blink_base import BlinkEstimatorBase
from rt_bene.blink_estimation_models_pytorch import BlinkEstimationModelResnet18
import os
import cv2
import torch
from torchvision import transforms


class BlinkEstimatorPytorch(BlinkEstimatorBase):

    def __init__(self, device_id_blink, model_files, threshold):
        super(BlinkEstimatorPytorch, self).__init__(device_id=device_id_blink, threshold=threshold)
        download_blink_models()
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(224, 224), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self._models = []
        for ckpt in model_files:
            _model = BlinkEstimationModelResnet18()
            _model.load_state_dict(torch.load(ckpt))
            _model.to(self.device_id)
            _model.eval()
            self._models.append(_model)

        tqdm.write('Loaded ' + str(len(self._models)) + ' model(s)')
        tqdm.write('Ready')

    def predict(self, left_eyes, right_eyes):
        transformed_left = torch.stack(left_eyes).to(self.device_id)
        transformed_right = torch.stack(right_eyes).to(self.device_id)

        result = [torch.sigmoid(model(transformed_left, transformed_right)).detach().cpu() for model in self._models]
        result = torch.stack(result, dim=1)
        result = torch.mean(result, dim=1).numpy()
        return result

    def inputs_from_images(self, left, right):
        return self._transform(left).to(self.device_id), self._transform(right).to(self.device_id)
