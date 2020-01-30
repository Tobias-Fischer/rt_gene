#!/usr/bin/env python

# from https://hackernoon.com/tf-serving-keras-mobilenetv2-632b8d92983c

from dataset_manager import RT_BENE

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

# from https://stackoverflow.com/questions/12116685/how-can-i-require-my-python-scripts-argument-to-be-a-float-between-0-0-1-0-usin
def restricted_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 1.0]"%(x,))
    return x

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_base", choices=['densenet121', 'resnet50', 'mobilenetv2'])
    parser.add_argument("model_path", help="target folder to save the models (auto-saved)")
    parser.add_argument("csv_subjects", help="")
    #parser.add_argument("dataset_imgs", help="")
    parser.add_argument("--use_weights", help="whether to use weights")
    parser.add_argument("--random_subset", type=restricted_float, help="")
    parser.add_argument("--batch_size", type=int, help="", default=64)
    parser.add_argument("--epochs", type=int, help="", default=8)
    parser.add_argument("--input_size", type=int, help="", default=96)

    args = parser.parse_args()
    model_base = args.model_base
    batch_size = args.batch_size
    epochs = args.epochs

    input_size = args.input_size

    if args.use_weights:
        weight_factor = 'use_weight'
    else:
        weight_factor = 'no_weight'

    dataset = RT_BENE(args.csv_subjects, input_size, args.random_subset)
    dataset.load()
    fold_infos = [([0, 1], [2], 'fold1'), ([0, 2], [1], 'fold2'), ([1, 2], [0], 'fold3')]

    # 3 folds training
    for subjects_train, subjects_test, fold_name in fold_infos:
        training_fold = dataset.get_folds(subjects_train)
        validation_fold = dataset.get_folds(subjects_test)

        positive = training_fold['positive']
        negative = training_fold['negative']

        print('Number of positive samples in training data: {} ({:.2f}% of total)'.format(positive, 100 * float(positive) / len(training_fold['y'])))

        metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        model, name = create_model(args.model_base, [input_size, input_size, 3], 1e-4, metrics)
        name = 'rt-bene_' + name + '_' + fold_name

        if args.use_weights:
            weight_for_0 = 1. / negative * (negative + positive)
            weight_for_1 = 1. / positive * (negative + positive)
            class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            class_weight = None

        best_name = args.model_path + name + '_best.h5'
        save_best = ModelCheckpoint(best_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
        auto_save = ModelCheckpoint(args.model_path + name + '_auto_{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False, period=1)

        # train the model
        model.fit(x=training_fold['x'], y=training_fold['y'], batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(validation_fold['x'], validation_fold['y']), callbacks=[save_best, auto_save], class_weight=class_weight)
        model, training_fold, validation_fold = None, None, None
        del model, training_fold, validation_fold
        gc.collect()
