#!/usr/bin/env python

from keras.applications.mobilenetv2 import MobileNetV2
from keras.applications.densenet import DenseNet121
from keras.applications.nasnet import NASNetMobile
from keras.applications.resnet50 import ResNet50

from keras.models import Model, load_model
from keras.layers import Conv2D, MaxPooling2D, Dense, Input, Dropout, Flatten, BatchNormalization, Add, Average, Maximum, Concatenate, Multiply, Activation, ReLU
from keras.optimizers import SGD, Adam
from keras.regularizers import l2

from keras import backend as K
import tensorflow as tf


# https://towardsdatascience.com/handling-imbalanced-datasets-in-deep-learning-f48407a0e758
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


# https://www.kaggle.com/rejpalcz/best-loss-function-for-f1-score-metric
def f1_loss(y_true, y_pred):
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


# https://www.scitepress.org/Papers/2017/61727/61727.pdf
def paper_cnn(input_shape, loss, lr, metrics, num_output_units=2):
    img_input = Input(shape=input_shape)
    c1 = Conv2D(4, (15, 15), strides=(2, 2), padding='same', activation='relu')(img_input)
    c2 = Conv2D(16, (11, 11), strides=(2, 2), activation='relu')(c1)
    pool1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c2)

    c3 = Conv2D(32, (7, 7), strides=(2, 2), padding='same', activation='relu')(pool1)
    pool2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c3)

    c4 = Conv2D(16, (3, 3), strides=(2, 2), padding='same', activation='relu')(pool2)
    pool3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(c4)

    flat = Flatten()(pool3)

    fc1 = Dense(32, activation='relu')(flat)
    fc2 = Dense(512, activation='relu')(fc1)

    if num_output_units == 1:
        output = Dense(num_output_units, activation='sigmoid')(fc2)
    else:
        output = Dense(num_output_units, activation='softmax')(fc2)

    cnn = Model(inputs=img_input, outputs=output)
    name = 'other_cnn'

    cnn.compile(loss=loss, optimizer=Adam(lr=lr), metrics=metrics)
    cnn.summary()
    return cnn, name


def load_existing_model(model_path, metrics=None):
    if metrics is None:
        metrics = {}
    return load_model(model_path, custom_objects=metrics)
#     return load_model(model_path)


def create_model_base(backbone, input_shape):
    """
    Get a model from keras.applications
    """

    assert backbone in ['mobilenetv2', 'densenet121', 'resnet50'], 'Wrong model_name: ' + backbone

    if backbone == 'mobilenetv2':
        base = MobileNetV2(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='avg')

    elif backbone == 'densenet121':
        base = DenseNet121(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='avg')

    elif backbone == 'resnet50':
        base = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=input_shape, pooling='avg')
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

    model_name = backbone + '_' + str(input_shape[0]) + '_avg_fc512_d06'

    return model, model_name

def create_model(backbone, input_shape, lr, metrics):
    base, name = create_model_base(backbone, input_shape)

    # define the 2 inputs (right and left eyes)
    right_input = Input(shape=input_shape)
    left_input = Input(shape=input_shape)

    # get the 2 outputs using shared layers
    out_right = base(right_input)
    out_left = base(left_input)

    # average
    merged = Average()([out_right, out_left])
    model = Model(inputs=[right_input, left_input], outputs=merged)

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=lr), metrics=metrics)
    model.summary()

    return model, name
