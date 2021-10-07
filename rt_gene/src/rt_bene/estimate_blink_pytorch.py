#! /usr/bin/env python

from tqdm import tqdm
from rt_gene.download_tools import download_blink_pytorch_models, md5
from rt_bene.estimate_blink_base import BlinkEstimatorBase
from rt_bene.blink_estimation_models_pytorch import BlinkEstimationModelResnet18, BlinkEstimationModelVGG16, BlinkEstimationModelVGG19, BlinkEstimationModelResnet50, BlinkEstimationModelDenseNet121
import os
import cv2
import torch
from torchvision import transforms

MODELS = {
    "resnet18": BlinkEstimationModelResnet18,
    "resnet50": BlinkEstimationModelResnet50,
    "vgg16": BlinkEstimationModelVGG16,
    "vgg19": BlinkEstimationModelVGG19,
    "densenet121": BlinkEstimationModelDenseNet121
}


class BlinkEstimatorPytorch(BlinkEstimatorBase):

    def __init__(self, device_id_blink, model_files, model_type, threshold, known_hashes=(
    "cde99055e3b6dcf9fae6b78191c0fd9b", "67339ceefcfec4b3b8b3d7ccb03fadfa", "e5de548b2a97162c5e655259463e4d23", "7c228fe7b95ce5960c4c5cae8f2d3a09", "0a0d2d066737b333737018d738de386f")):
        super(BlinkEstimatorPytorch, self).__init__(device_id=device_id_blink, threshold=threshold)
        download_blink_pytorch_models()

        assert model_type in MODELS.keys(), f"PyTorch backend only supports the following backends: [{','.join(MODELS.keys())}]"

        # check md5 hashes
        model_hashes = [md5(model) for model in model_files]
        correct = [1 for hash in model_hashes if hash not in known_hashes]
        if sum(correct) > 0:
            raise ImportError(
                "MD5 Hashes of supplied model_files do not match the known_hashes argument. You have probably not set "
                "the --models argument and therefore you are trying to use TensorFlow models. If you are training your "
                "own models, then please supply the md5sum hashes in the known_hashes argument. If you're not, "
                "then you're using old models. The newer models should have downloaded already so please update the "
                "estimate_blink.launch file that you've modified.")

        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self._transform = transforms.Compose([lambda x: cv2.resize(x, dsize=(60, 36), interpolation=cv2.INTER_CUBIC),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

        self._models = []
        for ckpt in model_files:
            _model = MODELS[model_type]()
            _model.load_state_dict(torch.load(ckpt))
            _model.to(self.device_id)
            _model.eval()
            self._models.append(_model)

        tqdm.write('Loaded ' + str(len(self._models)) + ' model(s)')
        tqdm.write('Ready')

    def predict(self, left_eyes, right_eyes):
        transformed_left = torch.stack(left_eyes).to(self.device_id)
        transformed_right = torch.stack(right_eyes).to(self.device_id)

        with torch.no_grad():
            result = [torch.sigmoid(model(transformed_left, transformed_right)).detach().cpu() for model in self._models]
            result = torch.stack(result, dim=1)
            result = torch.mean(result, dim=1).numpy()
            return result

    def inputs_from_images(self, left, right):
        return self._transform(left).to(self.device_id), self._transform(right).to(self.device_id)
