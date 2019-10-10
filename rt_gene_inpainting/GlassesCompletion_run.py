#!/usr/bin/env python

from GlassesCompletion import GlassesCompletion
from my_utils import ElapsedTimer

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.compat.v1.ConfigProto()
config.gpu_options.visible_device_list = "0"
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 1.0
set_session(tf.compat.v1.Session(config=config))

if __name__ == '__main__':
    dataset_folder_path = '/recordings_hdd/'
    subject = 's000'
    completion = GlassesCompletion(dataset_folder_path, subject)
    completion.image_completion_random_search(nIter=1000, GPU_ID=config.gpu_options.visible_device_list)

