#! /usr/bin/env python

import os
from argparse import ArgumentParser
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
from torchmetrics import Accuracy, F1, Precision, Recall, Specificity, MetricCollection
from rtbene_dataset import RTBENEH5Dataset


MODELS = {
    "resnet18": BlinkEstimationModelResnet18,
    "resnet50": BlinkEstimationModelResnet50,
    "vgg16": BlinkEstimationModelVGG16,
    "vgg19": BlinkEstimationModelVGG19,
    "densenet121": BlinkEstimationModelDenseNet121
}


class TrainRTBENE(pl.LightningModule):

    def __init__(self, hparams, train_subjects, validate_subjects, class_weights=None):
        super(TrainRTBENE, self).__init__()
        assert class_weights is not None, "Class Weights can't be None"

        self.model = MODELS[hparams.model_base]()
        self._criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([class_weights[1]]))
        self._train_subjects = train_subjects
        self._validate_subjects = validate_subjects
        self._metrics = MetricCollection([Accuracy(), F1(), Precision(), Recall(), Specificity()])
        self.save_hyperparameters(hparams, ignore=["train_subjects", "validate_subjects", "class_weights"])

    def forward(self, left_patch, right_patch):
        return self.model(left_patch, right_patch)

    def shared_step(self, batch, batch_idx):
        left, right, label = batch
        pred_blink = self.forward(left, right)
        loss = self._criterion(pred_blink, label)
        metrics = self._metrics(torch.sigmoid(pred_blink), label.type(torch.long))
        metrics["loss"] = loss

        return metrics, loss, label

    def training_step(self, batch, batch_idx):
        metrics, loss, _ = self.shared_step(batch, batch_idx)
        self.log_dict({"train_" + k: v for k, v in metrics.items()})
        return loss

    def validation_step(self, batch, batch_idx):
        metrics, loss, _ = self.shared_step(batch, batch_idx)
        self.log_dict({"val_" + k: v for k, v in metrics.items()})
        return loss

    def configure_optimizers(self):
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)

        optimizer = torch.optim.AdamW(params_to_update, lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.weight_decay)

        return [optimizer]

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--augment', action="store_true", dest="augment")
        parser.add_argument('--no_augment', action="store_false", dest="augment")
        parser.add_argument('--loss_fn', choices=["bce"], default="bce")
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--learning_rate', type=float, default=1e-3)
        parser.add_argument('--model_base', choices=MODELS.keys(), default=list(MODELS.keys())[0])
        parser.add_argument('--weight_decay', default=1e-2, type=float)
        return parser

    def train_dataloader(self):
        train_transforms = transforms.Compose([transforms.RandomResizedCrop(size=(36, 60), scale=(0.5, 1.3)),
                                               transforms.RandomPerspective(distortion_scale=0.2),
                                               transforms.RandomGrayscale(p=0.1),
                                               transforms.ColorJitter(brightness=0.5, hue=0.2, contrast=0.5,saturation=0.5),
                                               lambda x: x if np.random.random_sample() <= 0.1 else x.filter(ImageFilter.GaussianBlur(radius=2)),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        data_train = RTBENEH5Dataset(h5_pth=self.hparams.dataset, subject_list=self._train_subjects, transform=train_transforms, loader_desc="train")
        return DataLoader(data_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_io_workers, pin_memory=True)

    def val_dataloader(self):
        data_validate = RTBENEH5Dataset(h5_pth=self.hparams.dataset, subject_list=self._validate_subjects, loader_desc="valid")
        return DataLoader(data_validate, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_io_workers, pin_memory=True)


if __name__ == "__main__":
    from pytorch_lightning import Trainer
    from pytorch_lightning.plugins import DDPPlugin
    import psutil

    root_dir = os.path.dirname(os.path.realpath(__file__))

    root_parser = ArgumentParser(add_help=False)
    root_parser.add_argument('--gpu', type=int, default=-1, help='gpu to use, 1 for a single GPU, 3 for 3 GPUs, -1 for all (default: -1)')
    root_parser.add_argument('--dataset', type=str, default="/tmp/rtbene_dataset.hdf5")
    root_parser.add_argument('--dataset_name', type=str, choices=["rt_bene"], default="rt_bene")
    root_parser.add_argument('--num_io_workers', default=psutil.cpu_count(logical=False), type=int)
    root_parser.add_argument('--all_dataset', action='store_false', dest="k_fold_validation")
    root_parser.add_argument('--k_fold_validation', action="store_true", dest="k_fold_validation")
    root_parser.add_argument('--seed', type=int, default=0)
    root_parser.add_argument('--min_epochs', type=int, default=5, help="Number of Epochs to perform at a minimum")
    root_parser.add_argument('--max_epochs', type=int, default=20, help="Maximum number of epochs to perform; the trainer will Exit after.")
    root_parser.set_defaults(k_fold_validation=True)

    model_parser = TrainRTBENE.add_model_specific_args(root_parser, root_dir)
    hyperparams = model_parser.parse_args()

    pl.seed_everything(hyperparams.seed)

    train_subjects = []
    valid_subjects = []
    if hyperparams.dataset_name == "rt_bene":
        if hyperparams.k_fold_validation:
            # 6 is discarded
            train_subjects.append([1, 2, 8, 10])
            train_subjects.append([3, 4, 7, 9])
            train_subjects.append([5, 12, 13, 14])

            valid_subjects.append([0, 11, 15, 16])
            valid_subjects.append([0, 11, 15, 16])
            valid_subjects.append([0, 11, 15, 16])
        else:  # we want to train with the entire dataset
            print('Training on the whole dataset - do not use the trained model for evaluation purposes!')
            print('Validation dataset is a subject included in training...use at your own peril!')

            # 6 is discarded as per the paper
            train_subjects.append([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
            valid_subjects.append([7])
    else:
        raise NotImplementedError("No other dataset is currently implemented")

    for fold, (train_s, valid_s) in enumerate(zip(train_subjects, valid_subjects)):
        # this is a hack to get class weights, i'm sure there's a better way fo doing it but I can't think of it
        with h5py.File(hyperparams.dataset, mode="r") as _h5_f:
            class_weights = RTBENEH5Dataset.get_class_weights(h5_file=_h5_f, subject_list=train_s)

        model = TrainRTBENE(hparams=hyperparams, train_subjects=train_s, validate_subjects=valid_s, class_weights=class_weights)

        checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=3 if hyperparams.k_fold_validation else -1, filename=f'fold={fold}-' + '{epoch}-{val_loss:.3f}')

        # start training
        trainer = Trainer(gpus=hyperparams.gpu,
                          callbacks=[checkpoint_callback],
                          accelerator="ddp",
                          plugins=[DDPPlugin(find_unused_parameters=False)],
                          precision=32,
                          progress_bar_refresh_rate=1,
                          min_epochs=hyperparams.min_epochs,
                          max_epochs=hyperparams.max_epochs,
                          benchmark=True)
        trainer.fit(model)
