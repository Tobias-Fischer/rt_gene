#!/usr/bin/env python

# from https://hackernoon.com/tf-serving-keras-mobilenetv2-632b8d92983c

from dataset_manager import EyeBlinking8, RT_BENE, CEW, Talking, ResearcherNight

from blink_model_factory import create_model, load_existing_model
from keras.callbacks import ModelCheckpoint

from tqdm import tqdm

import gc

import argparse

import numpy as np

import keras_metrics
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
    parser.add_argument("ensemble_num")
    parser.add_argument("dataset_name", choices=['rt-bene', 'eyeblinking8'])
    parser.add_argument("model_path", help="path to the folder containing the trained models")
    parser.add_argument("eval_path", help="path to the folder containing the evaluation file")
    parser.add_argument("dataset_json", help="path to the dataset json file")
    parser.add_argument("dataset_imgs", help="path to the folder containing dataset images")
    parser.add_argument("--use_weights", help="whether to use weights")
    parser.add_argument("--random_subset", type=restricted_float, help="whether to use weights")

    args = parser.parse_args()
    model_base = args.model_base
    training = True
    testing = True
    batch_size = 64
    epochs = 8

    input_size = 96

    if args.use_weights:
        weight_factor = 'use_weight'
    else:
        weight_factor = 'no_weight'

    if args.dataset_name == 'rt-bene':
        dataset = RT_BENE(args.dataset_json, args.dataset_imgs)
        fold_infos = [([0, 1], [2], 'fold1'),
                 ([0, 2], [1], 'fold2'),
                 ([1, 2], [0], 'fold3')]

    elif args.dataset_name == 'eyeblinking8':
        dataset = EyeBlinking8(args.dataset_folder)
        fold_infos = [([1, 2, 10, 8, 4, 11], [3, 9], 'fold1'),
            ([2, 3, 10, 8, 9], [1, 4, 11], 'fold2'),
            ([1, 3, 9, 4, 11], [2, 10, 8], 'fold3')]

    folds, validations = dataset.load_folds()

    json_dict = {'model_name': model_base, 'dataset_name': args.dataset_name}

    # 3 folds
    for subjects_train, subjects_test, fold_name in fold_infos:
        if args.dataset_name == 'rt-bene':
            train_x, train_y, val_x, val_y, counts = dataset.load_pairs((input_size, input_size), [folds[subjects_train[0]], folds[subjects_train[1]]], [folds[subjects_test[0]]])
        else:
            if args.random_subset:
                train_x, train_y, counts = dataset.load_pairs((input_size, input_size), subjects=subjects_train, undersample=False, random_subset=args.random_subset)
            else:
                train_x, train_y, counts = dataset.load_pairs((input_size, input_size), subjects=subjects_train, undersample=False)
            val_x, val_y, _ = dataset.load_pairs((input_size, input_size), subjects=subjects_test, undersample=False)
        

        print('Number of positive samples in training data: {} ({:.2f}% of total)'.format(counts[1], 100 * float(counts[1]) / len(train_y)))

        metrics = ['accuracy', keras_metrics.binary_recall(), keras_metrics.binary_precision(), keras_metrics.binary_f1_score()]
        model, name = create_model(args.model_base, [input_size, input_size, 3], 1e-4, metrics)
        name = args.dataset_name + '_' + name + '_' + fold_name

        if args.use_weights:
            weight_for_0 = 1. / counts[0] * (counts[0] + counts[1])
            weight_for_1 = 1. / counts[1] * (counts[0] + counts[1])
            class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            class_weight = None

        best_name = args.model_path + name + '_best.h5'
        save_best = ModelCheckpoint(best_name, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
        auto_save = ModelCheckpoint(args.model_path + name + '_auto_{epoch:02d}.h5', verbose=1, save_best_only=False, save_weights_only=False, period=1)

        # train the model
        model.fit(x=train_x, y=train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(val_x, val_y), callbacks=[save_best, auto_save], class_weight=class_weight)
        model, train_x, train_y = None, None, None
        del model, train_x, train_y

        if testing:
            metrics = {'binary_f1_score': keras_metrics.binary_f1_score(),
               'binary_recall': keras_metrics.binary_recall(),
               'binary_precision': keras_metrics.binary_precision()}
            model = load_existing_model(best_name, metrics)
            predictions = []
            ground_truth = []
            start = time.time()
            preds = model.predict(val_x, verbose=1)

            for i in range(len(preds)):
                predictions.append(float(preds[i]))
                ground_truth.append(float(val_y[i]))
            inference_time = (time.time() - start) / len(val_y)
            json_dict[fold_name] = {'prediction_time': inference_time, 'predictions': predictions, 'ground_truth': ground_truth}
        model, val_x, val_y = None, None, None
        del model, val_x, val_y
        gc.collect()

    json_name = args.eval_path + 'eval_' + model_base + '_' + args.model_base
    json_name = json_name + '.json'

    with open(json_name, 'w') as jsonfile:
        json.dump(json_dict, jsonfile)
