#!/usr/bin/env python

# from https://hackernoon.com/tf-serving-keras-mobilenetv2-632b8d92983c

from dataset_manager import RTBeneDataset

from blink_model_factory import create_model
from tensorflow.keras.callbacks import ModelCheckpoint

from tqdm import tqdm

import gc

import argparse

import numpy as np

import time
import json

import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_base", choices=['densenet121', 'resnet50', 'mobilenetv2'])
    parser.add_argument("model_path", help="target folder to save the models (auto-saved)")
    parser.add_argument("csv_subjects", help="path to the dataset csv file")
    parser.add_argument("--use_weight_balancing", help="whether to use weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--input_size", type=tuple, help="input size of images", default=(96,96))

    args = parser.parse_args()
    model_base = args.model_base
    epochs = args.epochs
    batch_size = args.batch_size
    input_size = args.input_size
    use_weight_balancing = args.use_weight_balancing

    dataset = RTBeneDataset(args.csv_subjects, input_size)
    dataset.load()
    fold_infos = [([0, 1], 'fold1'), ([0, 2], 'fold2'), ([1, 2], 'fold3')]

    validation_set = dataset.get_validation_set()

    # 3 folds training
    for subjects_train, fold_name in fold_infos:
        training_fold = dataset.get_folds(subjects_train)
        positive = training_fold['positive']
        negative = training_fold['negative']

        print('Number of positive samples in training data: {} ({:.2f}% of total)'.format(positive, 100 * float(positive) / len(training_fold['y'])))

        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        model, name = create_model(args.model_base, [input_size[0], input_size[1], 3], 1e-4, metrics)
        name = 'rt-bene_' + name + '_' + fold_name

        if use_weight_balancing:
            weight_for_0 = 1. / negative * (negative + positive)
            weight_for_1 = 1. / positive * (negative + positive)
            class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            class_weight = None

        best_name = args.model_path + name + '_best.h5'
        save_best = ModelCheckpoint(best_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
        auto_save = ModelCheckpoint(args.model_path + name + '_auto_{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False, period=1)

        # train the model
        model.fit(x=training_fold['x'], y=training_fold['y'], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(validation_set['x'], validation_set['y']), callbacks=[save_best, auto_save], class_weight=class_weight)
        model, training_fold = None, None
        del model, training_fold
        gc.collect()
    validation_set = None
    del validation_set
