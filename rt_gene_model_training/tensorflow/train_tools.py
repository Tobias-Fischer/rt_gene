from __future__ import print_function, division

import math

import numpy as np
from tensorflow.keras import initializers
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, concatenate, \
    BatchNormalization, Activation
from tensorflow.keras.models import Model
from tqdm import tqdm


# from tqdm import tqdm_notebook as tqdm


def angle_loss(y_true, y_pred):
    from tensorflow.keras import backend as K
    return K.sum(K.square(y_pred - y_true), axis=-1)


class GeneratorsTwoEyes(object):
    def __init__(self, train_num, size_validation_set, batch_size, num_steps_epoch,
                 train_images_l, train_images_r, train_gazes, train_headposes, norm_type='subtract_vgg'):
        self.total_idx = np.arange(train_num)
        self.validation_idx = np.random.choice(train_num, size_validation_set, replace=False)
        self.train_idx = np.setdiff1d(self.total_idx, self.validation_idx)
        self.size_train_idx = len(self.train_idx)
        self.batch_size = batch_size
        self.num_steps_epoch = num_steps_epoch
        self.train_num = train_num
        self.size_validation_set = size_validation_set

        self.train_images_L = train_images_l
        self.train_images_R = train_images_r
        self.train_gazes = train_gazes
        self.train_headposes = train_headposes

        self.norm_type = norm_type

    def get_train_data(self):
        counter = -11

        while 1:
            if counter % self.num_steps_epoch == 0:
                counter = 1

                self.validation_idx = np.random.choice(self.train_num, self.size_validation_set, replace=False)
                self.train_idx = np.setdiff1d(self.total_idx, self.validation_idx)
                self.size_train_idx = len(self.train_idx)
            else:
                counter = counter + 1

            tr_img_l = np.zeros((self.batch_size, 36, 60, 3))
            tr_img_r = np.zeros((self.batch_size, 36, 60, 3))
            tr_gazes = np.zeros((self.batch_size, 2))
            tr_headposes = np.zeros((self.batch_size, 2))

            for i in range(self.batch_size):
                rand_index = np.random.randint(0, self.size_train_idx)
                index = self.train_idx[rand_index]
                tr_img_l[i, :] = get_normalized_image(self.train_images_L[index, :], norm_type=self.norm_type)
                tr_img_r[i, :] = get_normalized_image(self.train_images_R[index, :], norm_type=self.norm_type)
                tr_gazes[i, :] = self.train_gazes[index, :]
                tr_headposes[i, :] = self.train_headposes[index, :]

            yield (
                {'img_input_L': tr_img_l, 'img_input_R': tr_img_r, 'headpose_input': tr_headposes},
                {'pred_gaze': tr_gazes})

    def get_validation_data(self):
        while 1:
            tr_img_l = np.zeros((self.batch_size, 36, 60, 3))
            tr_img_r = np.zeros((self.batch_size, 36, 60, 3))
            tr_gazes = np.zeros((self.batch_size, 2))
            tr_headposes = np.zeros((self.batch_size, 2))

            for i in range(self.batch_size):
                rand_index = np.random.randint(0, self.size_validation_set)
                index = self.validation_idx[rand_index]
                tr_img_l[i, :] = get_normalized_image(self.train_images_L[index, :], norm_type=self.norm_type)
                tr_img_r[i, :] = get_normalized_image(self.train_images_R[index, :], norm_type=self.norm_type)
                tr_gazes[i, :] = self.train_gazes[index, :]
                tr_headposes[i, :] = self.train_headposes[index, :]

            yield (
                {'img_input_L': tr_img_l, 'img_input_R': tr_img_r, 'headpose_input': tr_headposes},
                {'pred_gaze': tr_gazes})


def get_normalized_image(raw_image, norm_type):
    reshaped_image = raw_image.copy().reshape(36, 60, 3, order='F').astype(np.float)

    if norm_type == 'subtract_vgg':
        reshaped_image[:, :, 0] = reshaped_image[:, :, 0] - 103.939
        reshaped_image[:, :, 1] = reshaped_image[:, :, 1] - 116.779
        reshaped_image[:, :, 2] = reshaped_image[:, :, 2] - 123.68
    elif norm_type == '-1to1':
        reshaped_image[:, :, 0] = reshaped_image[:, :, 0] / 127.5 - 1.0
        reshaped_image[:, :, 1] = reshaped_image[:, :, 1] / 127.5 - 1.0
        reshaped_image[:, :, 2] = reshaped_image[:, :, 2] / 127.5 - 1.0
    elif norm_type == '0to1':
        reshaped_image[:, :, 0] = reshaped_image[:, :, 0] / 255.0
        reshaped_image[:, :, 1] = reshaped_image[:, :, 1] / 255.0
        reshaped_image[:, :, 2] = reshaped_image[:, :, 2] / 255.0
    else:
        raise ValueError("wrong norm_type!:"+norm_type)

    return reshaped_image


def get_vgg_twoeyes(optimizer='adam', model_type='VGG19', fc1_size=1024, fc2_size=512, fc3_size=256):
    kern_init = initializers.glorot_normal()

    img_input_l = Input(shape=(36, 60, 3), name='img_input_L')
    img_input_r = Input(shape=(36, 60, 3), name='img_input_R')
    headpose_input = Input(shape=(2,), name='headpose_input')

    # create the base pre-trained model
    if model_type == 'VGG19':
        base_model_l = VGG19(input_tensor=img_input_l, weights='imagenet', include_top=False)
        base_model_r = VGG19(input_tensor=img_input_r, weights='imagenet', include_top=False)
    elif model_type == 'VGG16':
        base_model_l = VGG16(input_tensor=img_input_l, weights='imagenet', include_top=False)
        base_model_r = VGG16(input_tensor=img_input_r, weights='imagenet', include_top=False)
    else:
        raise Exception('Unknown model type in get_vgg_twoeyes')

    for layer_L in base_model_l.layers[1:]:
        layer_L.name = 'layer_L_' + layer_L.name
    for layer_R in base_model_r.layers[1:]:
        layer_R.name = 'layer_R_' + layer_R.name

    # add a global spatial average pooling layer
    x_l = base_model_l.output
    x_l = GlobalAveragePooling2D()(x_l)
    x_r = base_model_r.output
    x_r = GlobalAveragePooling2D()(x_r)

    # let's add a fully-connected layer
    x_l = Dense(fc1_size, kernel_initializer=kern_init)(x_l)
    x_l = BatchNormalization()(x_l)
    x_l = Activation('relu')(x_l)

    x_r = Dense(fc1_size, kernel_initializer=kern_init)(x_r)
    x_r = BatchNormalization()(x_r)
    x_r = Activation('relu')(x_r)

    x = concatenate([x_l, x_r])

    x = Dense(fc2_size, kernel_initializer=kern_init)(x)
    x = concatenate([x, headpose_input])

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(fc3_size, activation='relu', kernel_initializer=kern_init)(x)

    gaze_predictions = Dense(2, kernel_initializer=kern_init, name='pred_gaze')(x)

    # this is the model we will train
    model = Model(inputs=[img_input_l, img_input_r, headpose_input], outputs=gaze_predictions)
    model.compile(optimizer=optimizer, loss=angle_loss, metrics=['accuracy', accuracy_angle])
    return model


def get_train_test_data_twoeyes(files, label, do_shuffle=True):
    gazes = np.vstack([files[idx][label]['gazes'] for idx in range(len(files))])
    headposes = np.vstack([files[idx][label]['headposes'] for idx in range(len(files))])
    images_l = np.vstack([files[idx][label]['imagesL'] for idx in range(len(files))])
    images_r = np.vstack([files[idx][label]['imagesR'] for idx in range(len(files))])

    num_instances = images_l.shape[0]
    # dimension = images_l.shape[1]

    if do_shuffle:
        shuffle_idx = np.arange(num_instances)
        np.random.shuffle(shuffle_idx)
        images_l = images_l[shuffle_idx]
        images_r = images_r[shuffle_idx]
        gazes = gazes[shuffle_idx]
        headposes = headposes[shuffle_idx]

    print ("%s %s images loaded" % (num_instances, label))
    print ("shape of 'images' is %s" % (images_l.shape,))

    return images_l, images_r, gazes, headposes, num_instances


def get_test_data_twoeyes(test_images_l, test_images_r, norm_type='subtract_vgg'):
    test_num = test_images_l.shape[0]

    testimg_l = np.zeros((test_num, 36, 60, 3))
    testimg_r = np.zeros((test_num, 36, 60, 3))

    for i in tqdm(range(test_num)):
        testimg_l[i, :] = get_normalized_image(test_images_l[i, :], norm_type=norm_type)
        testimg_r[i, :] = get_normalized_image(test_images_r[i, :], norm_type=norm_type)

    return testimg_l, testimg_r


def get_train_info(train_num, validation_split, batch_size):
    num_steps_epoch = int(train_num*(1-validation_split))//batch_size
    if num_steps_epoch < 1:
        num_steps_epoch = 1
    num_steps_validation = int(train_num*validation_split)//batch_size
    if num_steps_validation < 1:
        num_steps_validation = 1
    size_validation_set = int(train_num*validation_split)

    print('Num steps epoch', num_steps_epoch)
    print('Num steps validation', num_steps_validation)
    print('Validation set size', size_validation_set)

    return num_steps_epoch, num_steps_validation, size_validation_set


def accuracy_angle(y_true, y_pred):
    from tensorflow.keras import backend as K
    import tensorflow as tf

    pred_x = -1*K.cos(y_pred[0])*K.sin(y_pred[1])
    pred_y = -1*K.sin(y_pred[0])
    pred_z = -1*K.cos(y_pred[0])*K.cos(y_pred[1])
    pred_norm = K.sqrt(pred_x*pred_x + pred_y*pred_y + pred_z*pred_z)

    true_x = -1*K.cos(y_true[0])*K.sin(y_true[1])
    true_y = -1*K.sin(y_true[0])
    true_z = -1*K.cos(y_true[0])*K.cos(y_true[1])
    true_norm = K.sqrt(true_x*true_x + true_y*true_y + true_z*true_z)

    angle_value = (pred_x*true_x + pred_y*true_y + pred_z*true_z) / (true_norm*pred_norm)
    K.clip(angle_value, -0.9999999999, 0.999999999)
    return (tf.acos(angle_value)*180.0)/math.pi


def accuracy_angle_2(y_true, y_pred):
    pred_x = -1*math.cos(y_pred[0])*math.sin(y_pred[1])
    pred_y = -1*math.sin(y_pred[0])
    pred_z = -1*math.cos(y_pred[0])*math.cos(y_pred[1])
    pred_norm = math.sqrt(pred_x*pred_x + pred_y*pred_y + pred_z*pred_z)

    true_x = -1*math.cos(y_true[0])*math.sin(y_true[1])
    true_y = -1*math.sin(y_true[0])
    true_z = -1*math.cos(y_true[0])*math.cos(y_true[1])
    true_norm = math.sqrt(true_x*true_x + true_y*true_y + true_z*true_z)

    angle_value = (pred_x*true_x + pred_y*true_y + pred_z*true_z) / (true_norm*pred_norm)
    np.clip(angle_value, -0.9999999999, 0.999999999)
    return math.degrees(math.acos(angle_value))


def accuracy_angle_openface(y_true, y_pred):
    pred_x = -1*math.cos(y_pred[0])*math.sin(y_pred[1])
    pred_y = -1*math.sin(y_pred[0])
    pred_z = -1*math.cos(y_pred[0])*math.cos(y_pred[1])
    pred = np.array([pred_x, pred_y, pred_z])
    pred = pred / np.linalg.norm(pred)
    
    true_x = -1*math.cos(y_true[0])*math.sin(y_true[1])
    true_y = -1*math.sin(y_true[0])
    true_z = -1*math.cos(y_true[0])*math.cos(y_true[1])
    gt = np.array([true_x, true_y, true_z])
    gt = gt / np.linalg.norm(gt)
    
    angle_err = np.rad2deg(np.arccos(np.dot(pred, gt)))
    return angle_err

