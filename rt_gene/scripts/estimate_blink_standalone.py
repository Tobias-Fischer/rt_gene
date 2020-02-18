#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

from os import listdir
import os
import argparse

import cv2
import numpy as np

from rt_bene.estimate_blink_base import BlinkEstimatorBase

script_path = os.path.dirname(os.path.realpath(__file__))


class BlinkEstimatorStandalone(BlinkEstimatorBase):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorStandalone, self).__init__("gpu", model_path, threshold, input_size)
        self.viz = viz


class BlinkEstimatorImagePair(BlinkEstimatorStandalone):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorImagePair, self).__init__(model_path, threshold, input_size, viz)

    def estimate(self, left_image_path, right_image_path):
        left_image = [self.load_img(left_image_path, False)]
        right_image = [self.load_img(right_image_path, True)]
        probs, blinks = self.predict(left_image, right_image)

        if self.viz:
            pair_img = np.concatenate((right_image[0], left_image[0]), axis=1)
            viz_img = self.overlay_prediction_over_img(pair_img, blinks[0])
            cv2.imshow('image pair visualisation', viz_img)
        return probs, blinks


class BlinkEstimatorFolderPair(BlinkEstimatorStandalone):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorFolderPair, self).__init__(model_path, threshold, input_size, viz)

    def estimate(self, left_folder_path, right_folder_path):
        left_images = []
        right_images = []

        for left_image_name in listdir(left_folder_path):
            left_image_path = left_folder_path + '/' + left_image_name
            left_images.append(self.load_img(left_image_path, False))

        for right_image_name in listdir(right_folder_path):
            right_image_path = right_folder_path + '/' + right_image_name
            right_images.append(self.load_img(right_image_path, True))

        probs, blinks = self.predict(left_images, right_images)
        if self.viz:
            for left_image, right_image, is_blinking in zip(left_images, right_images, blinks):
                pair_img = np.concatenate((right_image, left_image), axis=1)
                viz_img = self.overlay_prediction_over_img(pair_img, is_blinking)
                cv2.imshow('folder images visualisation', viz_img)
        return probs, blinks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate blink from image or folder pair.')
    parser.add_argument('left', type=str, help='Path to a left eye image or a directory containing left eye images')
    parser.add_argument('right', type=str, help='Path to a right eye image or a directory containing images right eye images')
    parser.add_argument('--model', nargs='+', type=str,
                        default=[os.path.join(script_path, '../model_nets/blink_model_1.h5')],
                        help='List of blink estimators')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold to determine weither the prediction is positive or negative')
    parser.add_argument('--vis_blink', type=bool, default=True,
                        help='Show the overlayed result on original image or not')

    args = parser.parse_args()
    left_path = args.left
    right_path = args.right

    if os.path.isfile(left_path) and os.path.isfile(right_path):
        blink_estimator = BlinkEstimatorImagePair(args.model, args.threshold, input_size=(96, 96), viz=args.vis_blink)
    elif os.path.isdir(left_path) and os.path.isdir(right_path):
        blink_estimator = BlinkEstimatorFolderPair(args.model, args.threshold, input_size=(96, 96), viz=args.vis_blink)
    else:
        raise Exception('Files / folders not found: Check that ' + left_path + ' and ' + right_path + 'exist')

    print(blink_estimator.estimate(left_path, right_path))
