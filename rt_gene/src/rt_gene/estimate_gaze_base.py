# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import os

import cv2
import numpy as np
import tensorflow
from keras.backend import set_session
from keras.engine.saving import load_model
from rt_gene.gaze_tools import accuracy_angle, angle_loss, get_endpoint
from tqdm import tqdm

import keras


class GazeEstimatorBase(object):
    """This class encapsulates a deep neural network for gaze estimation.

    It retrieves two image streams, one containing the left eye and another containing the right eye.
    It synchronizes these two images with the estimated head pose.
    The images are then converted in a suitable format, and a forward pass of the deep neural network
    results in the estimated gaze for this frame. The estimated gaze is then published in the (theta, phi) notation."""

    def __init__(self, device_id_gaze, model_files):
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = "8"
        tqdm.write("PyTorch using {} threads.".format(os.environ["OMP_NUM_THREADS"]))

        self.device_id_gazeestimation = device_id_gaze

        with tensorflow.device(self.device_id_gazeestimation):
            config = tensorflow.ConfigProto(inter_op_parallelism_threads=1,
                                            intra_op_parallelism_threads=1)
            if "gpu" in self.device_id_gazeestimation:
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 0.3
            config.log_device_placement = False
            self.sess = tensorflow.Session(config=config)
            set_session(self.sess)

        models = []
        img_input_l = keras.Input(shape=(36, 60, 3), name='img_input_L')
        img_input_r = keras.Input(shape=(36, 60, 3), name='img_input_R')
        headpose_input = keras.Input(shape=(2,), name='headpose_input')
        for model_file in model_files:
            tqdm.write('Load model ' + model_file)
            models.append(load_model(model_file,
                                     custom_objects={'accuracy_angle': accuracy_angle, 'angle_loss': angle_loss}))
            models[-1].name = "model_{}".format(len(models))

        if len(models) == 1:
            self._gaze_offset = 0.11
            self.ensemble_model = models[0]
        else:
            self._gaze_offset = 0.0
            tensors = [model([img_input_l, img_input_r, headpose_input]) for model in models]
            output_layer = keras.layers.average(tensors)
            self.ensemble_model = keras.Model(inputs=[img_input_l, img_input_r, headpose_input], outputs=output_layer)
        # noinspection PyProtectedMember
        self.ensemble_model._make_predict_function()

        tqdm.write('Loaded ' + str(len(models)) + ' models')

        self.graph = tensorflow.get_default_graph()

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        with self.graph.as_default():
            mean_prediction = self.ensemble_model.predict({'img_input_L': np.array(inference_input_left_list),
                                                           'img_input_R': np.array(inference_input_right_list),
                                                           'headpose_input': np.array(inference_headpose_list)})
            mean_prediction[:, 1] += self._gaze_offset
            return mean_prediction  # returns [subject : [gaze_pose]]

    @staticmethod
    def visualize_eye_result(eye_image, est_gaze):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(eye_image)

        center_x = output_image.shape[1] / 2
        center_y = output_image.shape[0] / 2

        endpoint_x, endpoint_y = get_endpoint(est_gaze[0], est_gaze[1], center_x, center_y, 50)

        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (255, 0, 0))
        return output_image

    @staticmethod
    def input_from_image(cv_image):
        """This method converts an eye_img_msg provided by the landmark estimator, and converts it to a format
        suitable for the gaze network."""
        currimg = cv_image.reshape(36, 60, 3, order='F')
        currimg = currimg.astype(np.float32)
        testimg = np.zeros((36, 60, 3))
        testimg[:, :, 0] = currimg[:, :, 0] - 103.939
        testimg[:, :, 1] = currimg[:, :, 1] - 116.779
        testimg[:, :, 2] = currimg[:, :, 2] - 123.68
        return testimg
