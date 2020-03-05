import os
from argparse import ArgumentParser

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image, ImageFilter
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from GazeAngleAccuracy import GazeAngleAccuracy
from RTGENEDataset import RTGENEH5Dataset
from RTGENEModels import RTGENEModelMobileNetV2, RTGENEModelResnet18, RTGENEModelResnet50, RTGENEModelVGG


class TrainRTGENE(pl.LightningModule):

    def __init__(self, hparams):
        super(TrainRTGENE, self).__init__()
        _models = {
            "vgg": RTGENEModelVGG,
            "mobilenet": RTGENEModelMobileNetV2,
            "resnet18": RTGENEModelResnet18,
            "resnet50": RTGENEModelResnet50
        }
        self._model = _models.get(hparams.model_base)()
        self._criterion = torch.nn.MSELoss(reduction="sum")
        self._angle_acc = GazeAngleAccuracy()
        self.hparams = hparams

    def forward(self, left_patch, right_patch, head_pose):
        return self._model(left_patch, right_patch, head_pose)

    def training_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels = batch

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        loss = self._criterion(angular_out, _gaze_labels)
        angle_acc = self._angle_acc(angular_out, _gaze_labels)
        tensorboard_logs = {'train_loss': loss, 'train_angle_acc': angle_acc}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        _left_patch, _right_patch, _headpose_label, _gaze_labels = batch

        angular_out = self.forward(_left_patch, _right_patch, _headpose_label)
        loss = self._criterion(angular_out, _gaze_labels)
        angle_acc = self._angle_acc(angular_out, _gaze_labels)

        return {'val_loss': loss, 'angle_acc': angle_acc}

    def validation_end(self, outputs):
        _losses = torch.stack([x['val_loss'] for x in outputs])
        _angles = np.array([x['angle_acc'] for x in outputs])
        tensorboard_logs = {'val_loss': _losses.mean(), 'val_angle_acc': np.mean(_angles), 'val_angle_std': np.std(_angles)}
        return {'val_loss': _losses.mean(), 'log': tensorboard_logs}

    def configure_optimizers(self):
        _params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                _params_to_update.append(param)

        _learning_rate = self.hparams.learning_rate
        _optimizer = torch.optim.Adam(_params_to_update, lr=_learning_rate, betas=(0.9, 0.95))
        return _optimizer

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        return parent_parser

    @pl.data_loader
    def train_dataloader(self):
        _train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0)),
                                                transforms.Resize((224, 224), Image.BICUBIC),
                                                transforms.RandomGrayscale(p=0.08),
                                                lambda x: x if np.random.random_sample() > 0.08 else x.filter(ImageFilter.GaussianBlur(radius=2)),
                                                lambda x: x if np.random.random_sample() > 0.0064 else x.filter(ImageFilter.GaussianBlur(radius=2)),
                                                transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _data_train = RTGENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"), subject_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
        return DataLoader(_data_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)

    @pl.data_loader
    def val_dataloader(self):
        _data_validate = RTGENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"), subject_list=[15, 16])
        return DataLoader(_data_validate, batch_size=self.hparams.batch_size, shuffle=True, num_workers=4)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    root_dir = os.path.dirname(os.path.realpath(__file__))

    _root_parser = ArgumentParser(add_help=False)

    # gpu args
    _root_parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    _root_parser.add_argument('--learning_rate', default=0.00075, type=float)
    _root_parser.add_argument('--model_base', choices=["vgg", "mobilenet", "resnet18", "resnet50"], default="vgg")
    _root_parser.add_argument('--hdf5_file', default=os.path.abspath("/home/ahmed/Documents/RT_GENE/dataset.hdf5"), type=str)
    _root_parser.add_argument('--save_dir', default=os.path.abspath(os.path.join(root_dir, '..', 'model_nets', 'rt_gene_pytorch_checkpoints')))
    _root_parser.add_argument('--batch_size', default=128, type=int)

    _model_parser = TrainRTGENE.add_model_specific_args(_root_parser, root_dir)
    _hyperparams = _model_parser.parse_args()

    _model = TrainRTGENE(hparams=_hyperparams)

    checkpoint_callback = ModelCheckpoint(filepath=_hyperparams.save_dir, monitor='val_loss', mode='min', verbose=True, save_top_k=3)  # save the best models

    # earlystopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=_hyperparams.gpus,
                      early_stop_callback=False,
                      checkpoint_callback=checkpoint_callback,
                      progress_bar_refresh_rate=1)
    trainer.fit(_model)
