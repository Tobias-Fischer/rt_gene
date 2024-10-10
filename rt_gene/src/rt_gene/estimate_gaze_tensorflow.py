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

        # Configure GPU settings if a GPU is specified
        if "gpu" in self.device_id_gazeestimation.lower():
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    for gpu in gpus:
                        # Enable memory growth to prevent TensorFlow from allocating all GPU memory
                        tf.config.experimental.set_memory_growth(gpu, True)
                except RuntimeError as e:
                    print(f"GPU configuration error: {e}")

        # Set device context
        self.device = tf.device(self.device_id_gazeestimation)
        with self.device:
            # Define input layers
            img_input_l = tf.keras.Input(shape=(36, 60, 3), name='img_input_L')
            img_input_r = tf.keras.Input(shape=(36, 60, 3), name='img_input_R')
            headpose_input = tf.keras.Input(shape=(2,), name='headpose_input')

            # Load models
            models = []
            for idx, model_file in enumerate(self.model_files, start=1):
                tqdm.write(f'Loading model {model_file}')
                model = tf.keras.models.load_model(model_file, compile=False)
                model._name = f"model_{idx}"
                models.append(model)

            # Create ensemble model
            if len(models) == 1:
                self.ensemble_model = models[0]
            elif len(models) > 1:
                # Collect outputs from all models
                outputs = [model([img_input_l, img_input_r, headpose_input]) for model in models]
                # Average the outputs
                averaged_output = tf.keras.layers.Average()(outputs)
                # Define the ensemble model
                self.ensemble_model = tf.keras.Model(
                    inputs=[img_input_l, img_input_r, headpose_input],
                    outputs=averaged_output,
                    name='ensemble_model'
                )
            else:
                raise ValueError("No models were loaded")

        tqdm.write(f'Loaded {len(models)} model(s)')

    def __del__(self):
        # No need to manually close sessions in TensorFlow 2.x
        pass

    def estimate_gaze_twoeyes(self, inference_input_left_list, inference_input_right_list, inference_headpose_list):
        """
        Estimate gaze using the ensemble model.

        Args:
            inference_input_left_list (list or np.ndarray): List of left eye images.
            inference_input_right_list (list or np.ndarray): List of right eye images.
            inference_headpose_list (list or np.ndarray): List of head pose data.

        Returns:
            np.ndarray: Gaze predictions with offset applied.
        """
        # Prepare inputs as a list matching the input layer order
        inputs = [
            np.array(inference_input_left_list),
            np.array(inference_input_right_list),
            np.array(inference_headpose_list)
        ]

        # Perform prediction
        mean_prediction = self.ensemble_model.predict(inputs)

        # Apply gaze offset
        mean_prediction[:, 1] += self._gaze_offset

        return mean_prediction  # returns [subject : [gaze_pose]]

    def input_from_image(self, cv_image):
        """
        Convert an eye image from the landmark estimator to the format suitable for the gaze network.

        Args:
            cv_image (np.ndarray): Eye image array.

        Returns:
            np.ndarray: Preprocessed image suitable for the model.
        """
        # Reshape and ensure correct data type
        currimg = cv_image.reshape(36, 60, 3, order='F').astype(float)

        # Initialize an array for the preprocessed image
        testimg = np.zeros((36, 60, 3))

        # Subtract mean values for each channel (BGR)
        testimg[:, :, 0] = currimg[:, :, 0] - 103.939
        testimg[:, :, 1] = currimg[:, :, 1] - 116.779
        testimg[:, :, 2] = currimg[:, :, 2] - 123.68

        return testimg
