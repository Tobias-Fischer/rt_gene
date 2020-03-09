#!/usr/bin/python

"""
VGG-16/VGG-19 architecture applied to RT-GENE Dataset
@ Tobias Fischer (t.fischer@imperial.ac.uk), Hyung Jin Chang (hj.chang@imperial.ac.uk)
"""

from __future__ import print_function, division, absolute_import

import argparse
import gc
import os

import h5py
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from train_tools import *

tf.compat.v1.disable_eager_execution()

path = '/recordings_hdd/mtcnn_twoeyes_inpainted_eccv/'


subjects_test_threefold = [
                           ['s001', 's002', 's008', 's010'],
                           ['s003', 's004', 's007', 's009'],
                           ['s005', 's006', 's011', 's012', 's013']
]
subjects_train_threefold = [
                            ['s003', 's004', 's007', 's009', 's005', 's006', 's011', 's012', 's013'],
                            ['s001', 's002', 's008', 's010', 's005', 's006', 's011', 's012', 's013'],
                            ['s001', 's002', 's008', 's010', 's003', 's004', 's007', 's009']
]

parser = argparse.ArgumentParser()
parser.add_argument("fc1_size", type=int)
parser.add_argument("fc2_size", type=int)
parser.add_argument("fc3_size", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("model_type", choices=['VGG16', 'VGG19'])
parser.add_argument("ensemble_num", type=int)
parser.add_argument("gpu_num", choices=['0', '1', '2', '3'])

args = parser.parse_args()

# Parameters
model_type = args.model_type
fc1_size = args.fc1_size
fc2_size = args.fc2_size
fc3_size = args.fc3_size

batch_size = args.batch_size
num_epochs = 4
validation_split = 0.05

suffix = 'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_'+str(args.ensemble_num)

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = args.gpu_num

for subjects_train, subjects_test in zip(subjects_train_threefold, subjects_test_threefold):
    print('subjects_test:', subjects_test)

    if os.path.isfile(path+"3Fold"+''.join(subjects_test)+suffix+"_01.h5") and \
       os.path.isfile(path+"3Fold"+''.join(subjects_test)+suffix+"_02.h5") and \
       os.path.isfile(path+"3Fold"+''.join(subjects_test)+suffix+"_03.h5") and \
       os.path.isfile(path+"3Fold"+''.join(subjects_test)+suffix+"_04.h5"):
        print('Skip training, model already exists: '+path+"3Fold"+''.join(subjects_test)+suffix+"_XX.h5")
        continue

    if os.path.isfile(path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_01.txt') and \
       os.path.isfile(path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_02.txt') and \
       os.path.isfile(path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_03.txt') and \
       os.path.isfile(path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_04.txt'):
        print('Skip training, model already evaluated: '+path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_XX.txt')
        continue

    K.clear_session()
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    train_file_names = [path+'/RT_GENE_train_'+subject+'.mat' for subject in subjects_train]
    train_files = [h5py.File(train_file_name) for train_file_name in train_file_names]

    train_images_L, train_images_R, train_gazes, train_headposes, train_num = get_train_test_data_twoeyes(train_files, 'train')

    num_steps_epoch, num_steps_validation, size_validation_set = get_train_info(train_num, validation_split, batch_size)

    generator = GeneratorsTwoEyes(train_num, size_validation_set, batch_size, num_steps_epoch,
                           train_images_L, train_images_R, train_gazes, train_headposes)

    adam = Adam(lr=0.00075, beta_1=0.9, beta_2=0.95)
    model = get_vgg_twoeyes(adam, model_type=model_type, fc1_size=fc1_size, fc2_size=fc2_size, fc3_size=fc3_size)

    checkpointer = ModelCheckpoint(filepath=path+"3Fold"+''.join(subjects_test)+suffix+"_{epoch:02d}.h5",
                                   verbose=1, save_best_only=False, period=1)

    history = model.fit_generator(generator.get_train_data(),
                                  steps_per_epoch=int(num_steps_epoch),
                                  epochs=num_epochs,
                                  use_multiprocessing=False,
                                  validation_data=generator.get_validation_data(),
                                  validation_steps=int(num_steps_validation),
                                  callbacks=[checkpointer])

    # model.save(path+"3Fold"+''.join(subjects_test)+suffix+".h5")

    for train_file in train_files:
        train_file.close()

    train_images_L, train_images_R, train_gazes, train_headposes = None, None, None, None
    model = None
    gc.collect()

