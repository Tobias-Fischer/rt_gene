# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from rt_gene.estimate_gaze_base import GazeEstimatorBase
from rt_gene.download_tools import download_gaze_tensorflow_models


class GazeEstimator(GazeEstimatorBase):
    def __init__(self, device_id_gaze, model_files):
        super(GazeEstimator, self).__init__(device_id_gaze, model_files)
        download_gaze_tensorflow_models()

        tf.compat.v1.disable_eager_execution()
        if 'output_all_intermediates' in dir(tf.compat.v1.experimental):
            tf.compat.v1.experimental.output_all_intermediates(True)

        with tf.device(self.device_id_gazeestimation):
            config = tf.compat.v1.ConfigProto(inter_op_parallelism_threads=1,
                                              intra_op_parallelism_threads=1)
            if "gpu" in self.device_id_gazeestimation:
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
            config.log_device_placement = False
            self.sess = tf.compat.v1.Session(config=config)
            tf.compat.v1.keras.backend.set_session(self.sess)

        models = []
        img_input_l = tf.keras.Input(shape=(36, 60, 3), name='img_input_L')
        img_input_r = tf.keras.Input(shape=(36, 60, 3), name='img_input_R')
        headpose_input = tf.keras.Input(shape=(2,), name='headpose_input')

        for model_file in self.model_files:
            tqdm.write('Load model ' + model_file)
            models.append(tf.keras.models.load_model(model_file, compile=False))
            # noinspection PyProtectedMember
            models[-1]._name = "model_{}".format(len(models))

        if len(models) == 1:
            self.ensemble_model = models[0]
        elif len(models) > 1:
            tensors = [model([img_input_l, img_input_r, headpose_input]) for model in models]
            output_layer = tf.keras.layers.average(tensors)
            self.ensemble_model = tf.keras.Model(inputs=[img_input_l, img_input_r, headpose_input], outputs=output_layer)
        else:
            raise ValueError("No models were loaded")
        # noinspection PyProtectedMember
        self.ensemble_model._make_predict_function()

        tqdm.write('Loaded ' + str(len(models)) + ' model(s)')

        self.graph = tf.compat.v1.get_default_graph()

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        with self.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.sess)
            mean_prediction = self.ensemble_model.predict({'img_input_L': np.array(inference_input_left_list),
                                                           'img_input_R': np.array(inference_input_right_list),
                                                           'headpose_input': np.array(inference_headpose_list)})
            mean_prediction[:, 1] += self._gaze_offset
            return mean_prediction  # returns [subject : [gaze_pose]]

    def input_from_image(self, cv_image):
        """This method converts an eye_img_msg provided by the landmark estimator, and converts it to a format
        suitable for the gaze network."""
        currimg = cv_image.reshape(36, 60, 3, order='F')
        currimg = currimg.astype(float)
        testimg = np.zeros((36, 60, 3))
        testimg[:, :, 0] = currimg[:, :, 0] - 103.939
        testimg[:, :, 1] = currimg[:, :, 1] - 116.779
        testimg[:, :, 2] = currimg[:, :, 2] - 123.68
        return testimg
