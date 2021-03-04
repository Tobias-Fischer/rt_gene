#! /usr/bin/env python

import os
from argparse import ArgumentParser
from functools import partial
import h5py
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from PIL import ImageFilter
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from rt_bene.blink_estimation_models_pytorch import BlinkEstimationModelResnet18, BlinkEstimationModelResnet50, \
    BlinkEstimationModelVGG16, BlinkEstimationModelVGG19, BlinkEstimationModelDenseNet121
from rtbene_dataset import RTBENEH5Dataset


class TrainRTBENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, class_weights=None):
        super(TrainRTBENE, self).__init__()
        assert class_weights is not None, "Class Weights can't be None"
        _loss_fn = {
            "bce": partial(torch.nn.BCEWithLogitsLoss, pos_weight=torch.Tensor([class_weights[1]]))
        }
        _param_num = {
            "bce": 1,
        }
        _models = {
            "resnet18": BlinkEstimationModelResnet18,
            "resnet50": BlinkEstimationModelResnet50,
            "vgg16": BlinkEstimationModelVGG16,
            "vgg19": BlinkEstimationModelVGG19,
            "densenet121": BlinkEstimationModelDenseNet121
        }
        self._model = _models.get(hparams.model_base)()
        self._criterion = _loss_fn.get(hparams.loss_fn)()
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self.hparams = hparams

    def forward(self, left_patch, right_patch):
        return self._model(left_patch, right_patch)

    def training_step(self, batch, batch_idx):
        _left, _right, _label = batch
        _pred_blink = self.forward(_left, _right)
        loss = self._criterion(_pred_blink, _label)
        self.log("train_loss", loss)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        _left, _right, _label = batch
        _pred_blink = self.forward(_left, _right)
        loss = self._criterion(_pred_blink, _label)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        _losses = torch.stack([x['val_loss'] for x in outputs])

        self.log("val_loss", _losses.mean())

    def configure_optimizers(self):
        _params_to_update = []
        for name, param in self._model.named_parameters():
            if param.requires_grad:
                _params_to_update.append(param)

        _optimizer = torch.optim.AdamW(_params_to_update, lr=self.hparams.learning_rate)
        _scheduler = torch.optim.lr_scheduler.StepLR(_optimizer, step_size=self.hparams.scheduler_step,
                                                     gamma=self.hparams.scheduler_gamma)

        return [_optimizer], [_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--augment', action="store_true", dest="augment")
        parser.add_argument('--no_augment', action="store_false", dest="augment")
        parser.add_argument('--loss_fn', choices=["bce"], default="bce")
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--model_base',
                            choices=["vgg16", "vgg19", "resnet18", "resnet50", "densenet121"],
                            default="densenet121")
        parser.add_argument('--scheduler_step', default=1, type=int)
        parser.add_argument('--scheduler_gamma', default=0.75, type=float)
        return parser

    def train_dataloader(self):
        _train_transforms = None
        if self.hparams.augment:
            _train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(224, 224), scale=(0.5, 1.3)),
                                                    transforms.RandomPerspective(distortion_scale=0.2),
                                                    transforms.RandomGrayscale(p=0.1),
                                                    transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5,
                                                                           saturation=0.5),
                                                    lambda x: x if np.random.random_sample() <= 0.1 else x.filter(
                                                        ImageFilter.GaussianBlur(radius=3)),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225])])

        _data_train = RTBENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"),
                                      subject_list=self._train_subjects, transform=_train_transforms,
                                      loader_desc="train")
        return DataLoader(_data_train, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        _data_validate = RTBENEH5Dataset(h5_file=h5py.File(self.hparams.hdf5_file, mode="r"),
                                         subject_list=self._validate_subjects, loader_desc="valid")
        return DataLoader(_data_validate, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_io_workers, pin_memory=True)


if __name__ == "__main__":
    from pytorch_lightning import Trainer

    root_dir = os.path.dirname(os.path.realpath(__file__))

    _root_parser = ArgumentParser(add_help=False)
    _root_parser.add_argument('--gpu', type=int, default=1,
                              help='gpu to use, can be repeated for mutiple gpus i.e. --gpu 1 --gpu 2', action="append")
    _root_parser.add_argument('--hdf5_file', type=str,
                              default=os.path.abspath(os.path.join(root_dir, "../../RT_BENE/rtbene_dataset.hdf5")))
    _root_parser.add_argument('--dataset', type=str, choices=["rt_bene"], default="rt_bene")
    _root_parser.add_argument('--save_dir', type=str, default=os.path.abspath(
        os.path.join(root_dir, '../../rt_bene_model_training/pytorch/checkpoints')))
    _root_parser.add_argument('--benchmark', action='store_true', dest="benchmark")
    _root_parser.add_argument('--no-benchmark', action='store_false', dest="benchmark")
    _root_parser.add_argument('--num_io_workers', default=4, type=int)
    _root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    _root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    _root_parser.add_argument('--accumulate_grad_batches', default=1, type=int)
    _root_parser.add_argument('--seed', type=int, default=0)
    _root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    _root_parser.add_argument('--max_epochs', type=int, default=20,
                              help="Maximum number of epochs to perform; the trainer will Exit after.")
    _root_parser.set_defaults(benchmark=False)
    _root_parser.set_defaults(augment=True)

    _model_parser = TrainRTBENE.add_model_specific_args(_root_parser, root_dir)
    _hyperparams = _model_parser.parse_args()

    pl.seed_everything(_hyperparams.seed)

    _train_subjects = []
    _valid_subjects = []
    if _hyperparams.dataset == "rt_bene":
        if _hyperparams.k_fold_validation:
            # 6 is discarded
            _train_subjects.append([1, 2, 8, 10])
            _train_subjects.append([3, 4, 7, 9])
            _train_subjects.append([5, 12, 13, 14])

            _valid_subjects.append([0, 11, 15, 16])
            _valid_subjects.append([0, 11, 15, 16])
            _valid_subjects.append([0, 11, 15, 16])
        else:  # we want to train with the entire dataset
            print('Training on the whole dataset - do not use the trained model for evaluation purposes!')
            print('Validation dataset is a subject included in training...use at your own peril!')

            # 6 is discarded as per the paper
            _train_subjects.append([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            _valid_subjects.append([7])
    else:
        raise NotImplementedError("No other dataset is currently implemented")

    for fold, (train_s, valid_s) in enumerate(zip(_train_subjects, _valid_subjects)):
        # this is a hack to get class weights, i'm sure there's a better way fo doing it but I can't think of it
        with h5py.File(_hyperparams.hdf5_file, mode="r") as _h5_f:
            _class_weights = RTBENEH5Dataset.get_class_weights(h5_file=_h5_f, subject_list=train_s)

        _model = TrainRTBENE(hparams=_hyperparams,
                             train_subjects=train_s,
                             validate_subjects=valid_s,
                             class_weights=_class_weights)

        checkpoint_callback = ModelCheckpoint(dirpath=_hyperparams.save_dir,
                                              monitor='val_loss',
                                              save_top_k=3 if _hyperparams.k_fold_validation else -1,
                                              filename=f'fold={fold}-' + '{epoch}-{val_loss:.3f}')

        # start training
        trainer = Trainer(gpus=_hyperparams.gpu,
                          callbacks=[checkpoint_callback],
                          precision=32,
                          progress_bar_refresh_rate=1,
                          min_epochs=_hyperparams.min_epochs,
                          max_epochs=_hyperparams.max_epochs,
                          accumulate_grad_batches=_hyperparams.accumulate_grad_batches,
                          benchmark=_hyperparams.benchmark)
        trainer.fit(_model)
