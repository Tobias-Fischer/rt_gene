#!/usr/bin/env python

import gc
import argparse

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Average, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from dataset_manager import RTBeneDataset

tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def create_model_base(backbone, input_shape):
    if backbone == 'mobilenetv2':
        base = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_tensor=None,
                                                 input_shape=input_shape, pooling='avg')

    elif backbone == 'densenet121':
        base = tf.keras.applications.DenseNet121(include_top=False, weights='imagenet', input_tensor=None,
                                                 input_shape=input_shape, pooling='avg')

    elif backbone == 'resnet50':
        base = tf.keras.applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                              input_shape=input_shape, pooling='avg')
    else:
        raise Exception('Wrong backbone')

    for layer in base.layers:
        layer.trainable = True

    main_input = Input(shape=input_shape)
    main_output = base(main_input)

    final_fc_layer = Dense(512, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))(main_output)
    final_fc_layer = BatchNormalization(epsilon=1e-3, momentum=0.999)(final_fc_layer)
    final_fc_layer = ReLU(6.)(final_fc_layer)
    final_fc_layer = Dropout(0.6)(final_fc_layer)
    output_tensor = Dense(1, activation='sigmoid')(final_fc_layer)  # probability

    model = Model(inputs=main_input, outputs=output_tensor)

    return model


def create_model(backbone, input_shape, lr, metrics):
    base = create_model_base(backbone, input_shape)

    # define the 2 inputs (left and right eyes)
    left_input = Input(shape=input_shape)
    right_input = Input(shape=input_shape)

    # get the 2 outputs using shared layers
    out_left = base(left_input)
    out_right = base(right_input)

    # average the predictions
    merged = Average()([out_left, out_right])
    model = Model(inputs=[right_input, left_input], outputs=merged)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=metrics)
    model.summary()

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_base", choices=['densenet121', 'resnet50', 'mobilenetv2'])
    parser.add_argument("model_path", help="target folder to save the models (auto-saved)")
    parser.add_argument("csv_subject_list", help="path to the dataset csv file")
    parser.add_argument("--use_weight_balancing", help="whether to use weights")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--input_size", type=tuple, help="input size of images", default=(96, 96))

    args = parser.parse_args()
    model_base = args.model_base
    epochs = args.epochs
    batch_size = args.batch_size
    input_size = args.input_size
    use_weight_balancing = args.use_weight_balancing

    dataset = RTBeneDataset(args.csv_subject_list, input_size)
    fold_infos = [
        ([0, 1], 'fold1'),
        ([0, 2], 'fold2'),
        ([1, 2], 'fold3')
    ]

    validation_set = dataset.get_validation_data()

    # 3 folds training
    for subjects_train, fold_name in fold_infos:
        training_fold = dataset.get_training_data(subjects_train)
        positive = training_fold['positive']
        negative = training_fold['negative']

        print('Number of positive samples in training data: {} ({:.2f}% of total)'.
              format(positive, 100 * float(positive) / len(training_fold['y'])))

        model_metrics = [tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.Recall(), tf.keras.metrics.Precision()]
        model_instance = create_model(args.model_base, [input_size[0], input_size[1], 3], 1e-4, model_metrics)
        name = 'rt-bene_' + args.model_base + '_' + fold_name

        if use_weight_balancing:
            weight_for_0 = 1. / negative * (negative + positive)
            weight_for_1 = 1. / positive * (negative + positive)
            class_weight = {0: weight_for_0, 1: weight_for_1}
        else:
            class_weight = None

        best_name = args.model_path + name + '_best.h5'
        save_best = ModelCheckpoint(best_name, monitor='val_loss', verbose=1, save_best_only=True,
                                    save_weights_only=False, mode='min', period=1)
        auto_save = ModelCheckpoint(args.model_path + name + '_auto_{epoch:02d}.h5', verbose=1,
                                    save_best_only=False, save_weights_only=False, period=1)

        # train the model
        model_instance.fit(x=training_fold['x'], y=training_fold['y'], batch_size=batch_size, epochs=epochs, verbose=1,
                           validation_data=(validation_set['x'], validation_set['y']), callbacks=[save_best, auto_save],
                           class_weight=class_weight)
        model_instance, training_fold = None, None
        del model_instance, training_fold
        gc.collect()
    validation_set = None
    del validation_set
