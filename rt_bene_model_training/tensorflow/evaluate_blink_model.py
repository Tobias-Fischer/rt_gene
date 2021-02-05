#!/usr/bin/env python

import gc

import tensorflow as tf
from tensorflow.keras.models import load_model

from sklearn.metrics import confusion_matrix, roc_curve, auc, average_precision_score

import numpy as np

tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


fold_infos = {
    'fold1': [2],
    'fold2': [1],
    'fold3': [0],
    'all': [2, 1, 0]
}

model_metrics = [tf.keras.metrics.BinaryAccuracy()]


def estimate_metrics(testing_fold, model_instance):
    threshold = 0.5
    p = model_instance.predict(x=testing_fold['x'], verbose=0)
    p = p >= threshold
    matrix = confusion_matrix(testing_fold['y'], p)
    ap = average_precision_score(testing_fold['y'], p)
    fpr, tpr, thresholds = roc_curve(testing_fold['y'], p)
    roc = auc(fpr, tpr)
    return matrix, ap, roc


def get_metrics_from_matrix(matrix):
    tp, tn, fp, fn = matrix[1, 1], matrix[0, 0], matrix[0, 1], matrix[1, 0]
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1score = 2. * (precision * recall) / (precision + recall)
    return precision, recall, f1score


def threefold_evaluation(dataset, model_paths_fold1, model_paths_fold2, model_paths_fold3, input_size):
    folds = ['fold1', 'fold2', 'fold3']
    aps = []
    rocs = []
    recalls = []
    precisions = []
    f1scores = []
    models = []
    
    for fold_to_eval_on, model_paths in zip(folds, [model_paths_fold1, model_paths_fold2, model_paths_fold3]):
        if len(model_paths_fold1) > 1:
            models = [load_model(model_path, compile=False) for model_path in model_paths]
            img_input_l = tf.keras.Input(shape=input_size, name='img_input_L')
            img_input_r = tf.keras.Input(shape=input_size, name='img_input_R')
            tensors = [model([img_input_r, img_input_l]) for model in models]
            output_layer = tf.keras.layers.average(tensors)
            model_instance = tf.keras.Model(inputs=[img_input_r, img_input_l], outputs=output_layer)
        else:
            model_instance = load_model(model_paths[0])
        model_instance.compile()

        testing_fold = dataset.get_training_data(fold_infos[fold_to_eval_on])  # get the testing fold subjects

        matrix, ap, roc = estimate_metrics(testing_fold, model_instance)
        aps.append(ap)
        rocs.append(roc)
        precision, recall, f1score = get_metrics_from_matrix(matrix)
        recalls.append(recall)
        precisions.append(precision)
        f1scores.append(f1score)

        del model_instance, testing_fold
        # noinspection PyUnusedLocal
        for model in models:
            del model
        gc.collect()

    evaluation = {'AP': {}, 'ROC': {}, 'precision': {}, 'recall': {}, 'f1score': {}}
    evaluation['AP']['avg'] = np.mean(np.array(aps))
    evaluation['AP']['std'] = np.std(np.array(aps))
    evaluation['ROC']['avg'] = np.mean(np.array(rocs))
    evaluation['ROC']['std'] = np.std(np.array(rocs))
    evaluation['precision']['avg'] = np.mean(np.array(precisions))
    evaluation['precision']['std'] = np.std(np.array(precisions))
    evaluation['recall']['avg'] = np.mean(np.array(recalls))
    evaluation['recall']['std'] = np.std(np.array(recalls))
    evaluation['f1score']['avg'] = np.mean(np.array(f1scores))
    evaluation['f1score']['std'] = np.std(np.array(f1scores))
    return evaluation
