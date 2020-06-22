from __future__ import print_function, division, absolute_import

import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from rt_gene.download_tools import download_blink_models

BLINK_COLOR = (0, 0, 255)
NO_BLINK_COLOR = (0, 255, 0)
OVERLAY_SIZE = 5


class BlinkEstimatorBase(object):
    def __init__(self, device_id_blink, model_files, threshold, input_size):
        download_blink_models()
        self.device_id_blink = device_id_blink
        self.threshold = threshold
        self.input_size = input_size

        tf.compat.v1.disable_eager_execution()

        with tf.device(self.device_id_blink):
            config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1,
                                              intra_op_parallelism_threads=1)
            if "gpu" in self.device_id_blink:
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
            config.log_device_placement = False
            self.sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(self.sess)

        if not isinstance(model_files, list):
            model_files = [model_files]

        models = []
        for model_path in model_files:
            tqdm.write('Load model ' + model_path)
            models.append(tf.keras.models.load_model(model_path, compile=False))
            # noinspection PyProtectedMember
            models[-1]._name = "model_{}".format(len(models))

        img_input_l = tf.keras.Input(shape=input_size+(3,), name='img_input_L')
        img_input_r = tf.keras.Input(shape=input_size+(3,), name='img_input_R')

        if len(models) == 1:
            self.model = models[0]
        else:
            tensors = [model([img_input_r, img_input_l]) for model in models]
            output_layer = tf.keras.layers.average(tensors)
            self.model = tf.keras.Model(inputs=[img_input_r, img_input_l], outputs=output_layer)

        # noinspection PyProtectedMember
        self.model._make_predict_function()
        self.graph = tf.compat.v1.get_default_graph()

        self.predict(np.zeros((1,)+input_size+(3,)), np.zeros((1,)+input_size+(3,)))

        tqdm.write('Loaded ' + str(len(models)) + ' model(s)')
        tqdm.write('Ready')

    def resize_img(self, img):
        return cv2.resize(img, dsize=self.input_size, interpolation=cv2.INTER_CUBIC)

    def predict(self, left_eyes, right_eyes):
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            x = [np.array(right_eyes), np.array(left_eyes)]  # model expects this order!
            p = self.model.predict(x, verbose=0)
            blinks = p >= self.threshold
            return p, blinks

    def overlay_prediction_over_img(self, img, p, border_size=OVERLAY_SIZE):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        if p > self.threshold:
            cv2.rectangle(img_copy, (0, 0), (w, h), BLINK_COLOR, border_size)
        else:
            cv2.rectangle(img_copy, (0, 0), (w, h), NO_BLINK_COLOR, border_size)
        return img_copy
