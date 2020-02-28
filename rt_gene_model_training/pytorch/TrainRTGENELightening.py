import os
import random
from argparse import ArgumentParser

import numpy as np
import pytorch_lightning as pl
import torch
from PIL import ImageFilter
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from torchvision import transforms

from GazeAngleAccuracy import GazeAngleAccuracy
from RTGENEFileDataset import RTGENEDataset
from RTGENEModel_Mobilenetv2 import RTGENEModelMobileNetV2
from RTGENEModel_Resnet import RTGENEModelResnet18, RTGENEModelResnet50
from RTGENEModel_VGG16 import RTGENEModelVGG


class TrainRTGENE(pl.LightningModule):

    def tng_dataloader(self):
        pass

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
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_angle = np.array([x['angle_acc'] for x in outputs])
        avg_angle = np.mean(avg_angle)
        tensorboard_logs = {'val_loss': avg_loss, 'val_angle_acc': avg_angle}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

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
        _transform = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.0)),  # don't crop too much as our eyes are already bounded
                                         transforms.RandomGrayscale(),
                                         # transforms.RandomRotation(10, resample=Image.BILINEAR),
                                         transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                         lambda x: x.filter(ImageFilter.GaussianBlur(radius=5 if random.random() >= 0.5 else 0)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _data_train = RTGENEDataset(root_path=self.hparams.data_root, subject_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], transform=_transform)
        return DataLoader(_data_train, batch_size=self.hparams.batch_size, shuffle=True)

    @pl.data_loader
    def val_dataloader(self):
        _data_validate = RTGENEDataset(root_path=self.hparams.data_root, subject_list=[15, 16])
        return DataLoader(_data_validate, batch_size=self.hparams.batch_size, shuffle=True)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    root_dir = os.path.dirname(os.path.realpath(__file__))

    _root_parser = ArgumentParser(add_help=False)

    # gpu args
    _root_parser.add_argument('--gpus', type=int, default=1, help='how many gpus')
    _root_parser.add_argument('--learning_rate', default=0.001, type=float)
    _root_parser.add_argument('--model_base', choices=["vgg", "mobilenet", "resnet18", "resnet50"], default="vgg")
    _root_parser.add_argument('--data_root', default=os.path.abspath("/home/ahmed/Documents/RT_GENE/"), type=str)
    _root_parser.add_argument('--save_dir', default=os.path.abspath(os.path.join(root_dir, '..', 'model_nets', 'rt_gene_pytorch_checkpoints')))
    _root_parser.add_argument('--batch_size', default=128, type=int)

    _model_parser = TrainRTGENE.add_model_specific_args(_root_parser, root_dir)
    _hyperparams = _model_parser.parse_args()

    _model = TrainRTGENE(hparams=_hyperparams)

    checkpoint_callback = ModelCheckpoint(filepath=_hyperparams.save_dir, monitor='val_loss', mode='min', verbose=True, save_top_k=2)  # save all the models

    earlystopping_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True)

    # most basic trainer, uses good defaults
    trainer = Trainer(gpus=_hyperparams.gpus, early_stop_callback=earlystopping_callback, checkpoint_callback=checkpoint_callback, progress_bar_refresh_rate=1)
    trainer.fit(_model)
