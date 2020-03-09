#!/usr/bin/python

"""
VGG-16/VGG-19 architecture applied to RT-GENE Dataset
@ Tobias Fischer (t.fischer@imperial.ac.uk), Hyung Jin Chang (hj.chang@imperial.ac.uk)
"""

from __future__ import print_function, division, absolute_import

import argparse
import gc
import os.path

import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model
from train_tools import *

tf.compat.v1.disable_eager_execution()

parser = argparse.ArgumentParser()
parser.add_argument("fc1_size", type=int)
parser.add_argument("fc2_size", type=int)
parser.add_argument("fc3_size", type=int)
parser.add_argument("batch_size", type=int)
parser.add_argument("model_type", choices=['VGG16', 'VGG19'])
parser.add_argument("epoch", choices=['01', '02', '03', '04'])
parser.add_argument("gpu_num", choices=['0', '1', '2', '3'])

args = parser.parse_args()

model_type = args.model_type
fc1_size = args.fc1_size
fc2_size = args.fc2_size
fc3_size = args.fc3_size
batch_size = args.batch_size
epoch = args.epoch


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = args.gpu_num
# config.gpu_options.per_process_gpu_memory_fraction = 0.7


evaluate_train = True

model_path = '/recordings_hdd/mtcnn_twoeyes_inpainted_eccv/'

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

model_prefixes = ['', '', '', '']
model_suffixes = ['eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_1_'+epoch,
                  'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_2_'+epoch,
                  'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_3_'+epoch,
                  'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_4_'+epoch,]

assert len(model_prefixes) == len(model_suffixes)

all_models_exist = True
for subjects_train, subjects_test in zip(subjects_train_threefold, subjects_test_threefold):
    for prefix, suffix in zip(model_prefixes, model_suffixes):
        if not os.path.isfile(model_path+"3Fold"+prefix+''.join(subjects_test)+suffix+".h5"):
            print('File does not exist', model_path+"3Fold"+prefix+''.join(subjects_test)+suffix+".h5")
            all_models_exist = False
            break

if os.path.isfile(model_path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_'+epoch+'.txt'):
    model_evaluated = True
else:
    model_evaluated = False

assert all_models_exist == True and model_evaluated == False


scores_list = []
scores_list_models = []
for model_num in range(0, len(model_suffixes)):
    scores_list_models.append([])

for subjects_train, subjects_test in zip(subjects_train_threefold, subjects_test_threefold):
    print('subjects_test:', subjects_test)
    K.clear_session()
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    models = []
    for prefix, suffix in zip(model_prefixes, model_suffixes):
        models.append(load_model(model_path+"3Fold"+prefix+''.join(subjects_test)+suffix+".h5", custom_objects={'accuracy_angle': accuracy_angle, 'angle_loss': angle_loss}))

    # Parse Test Data
    test_file_names = [model_path+'/RT_GENE_test_'+subject+'.mat' for subject in subjects_test]
    test_files = [h5py.File(test_file_name) for test_file_name in test_file_names]

    if evaluate_train:
        train_file_names = [model_path+'/RT_GENE_train_'+subject+'.mat' for subject in subjects_test]
        train_files = [h5py.File(train_file_name) for train_file_name in train_file_names]

    test_images_L_1, test_images_R_1, test_gazes_1, test_headposes_1, test_num_1 = get_train_test_data_twoeyes(test_files, 'test', do_shuffle=False)
    if evaluate_train:
        test_images_L_2, test_images_R_2, test_gazes_2, test_headposes_2, test_num_2 = get_train_test_data_twoeyes(train_files, 'train', do_shuffle=False)

    if evaluate_train:
        test_num_total = test_num_1 + test_num_2
        test_images_L_total = np.vstack((test_images_L_1, test_images_L_2))
        test_images_R_total = np.vstack((test_images_R_1, test_images_R_2))
        test_gazes_total = np.vstack((test_gazes_1, test_gazes_2))
        test_headposes_total = np.vstack((test_headposes_1, test_headposes_2))
    else:
        test_num_total = test_num_1
        test_images_L_total, test_images_R_total = test_images_L_1, test_images_R_1
        test_gazes_total, test_headposes_total = test_gazes_1, test_headposes_1

    testimg_L, testimg_R = get_test_data_twoeyes(test_images_L_total, test_images_R_total)

    est_gazes = []
    for model in models:
        est_gazes.append(model.predict({'img_input_L': testimg_L, 'img_input_R': testimg_R, 'headpose_input':test_headposes_total}, verbose=1))

    total_errors = [0.0] * len(models)
    total_errorc = 0.0
    for i in range(test_num_total):
        errors = [0.0] * len(models)
        combined_gaze = [0.0, 0.0]
        for model_num in range(len(models)):
            errors[model_num] = accuracy_angle_openface(est_gazes[model_num][i], test_gazes_total[i])
            total_errors[model_num] += errors[model_num]
            combined_gaze[0] += est_gazes[model_num][i][0]
            combined_gaze[1] += est_gazes[model_num][i][1]
        combined_gaze[0] = combined_gaze[0] / len(models)
        combined_gaze[1] = combined_gaze[1] / len(models)
        errorc = accuracy_angle_openface(combined_gaze, test_gazes_total[i])
        total_errorc += errorc

    for model_num in range(len(models)):
        total_errors[model_num] = total_errors[model_num] / test_num_total
    total_errorc = total_errorc / test_num_total
    print('\n')

    for model_num in range(len(models)):
        print('Error model', model_num, ':', total_errors[model_num])
        scores_list_models[model_num].append(total_errors[model_num])

    print('Error combined', total_errorc)
    scores_list.append(total_errorc)

    for test_file in test_files:
        test_file.close()
    testimg_L, testimg_R = None, None 
    test_images_L_1, test_images_R_1, test_gazes_1, test_headposes_1 = None, None, None, None
    if evaluate_train:
        test_images_L_2, test_images_R_2, test_gazes_2, test_headposes_2 = None, None, None, None
    test_images_L_total, test_images_R_total, test_gazes_total, test_headposes_total = None, None, None, None

    model = None

    gc.collect()
    K.clear_session()


with open(model_path+'eccv_'+model_type+'_'+str(fc1_size)+'_'+str(fc2_size)+'_'+str(fc3_size)+'_'+str(batch_size)+'_'+epoch+'.txt', "w") as f:
    print(scores_list)

    print('Ensemble: ' + str(np.mean(scores_list)) + ' +- ' + str(np.std(scores_list)))
    f.write('Ensemble: ' + str(np.mean(scores_list)) + ' +- ' + str(np.std(scores_list)) + '\n')
    for idx, score_model in enumerate(scores_list_models):
        print('Model ' + str(idx) + ': ' + str(np.mean(score_model)) + ' +- ' + str(np.std(score_model)))
        f.write('Model ' + str(idx) + ': ' + str(np.mean(score_model)) + ' +- ' + str(np.std(score_model)) + '\n')

