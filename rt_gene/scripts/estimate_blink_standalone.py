#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

from os import listdir
from rt_bene.estimate_blink_base import BlinkEstimatorBase
import sys
import os
import argparse

import cv2
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))

class BlinkEstimatorStandalone(BlinkEstimatorBase):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorStandalone, self).__init__("gpu", model_path, threshold, input_size)
        self.viz = viz
            

class BlinkEstimatorImagePair(BlinkEstimatorStandalone):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorImagePair, self).__init__(model_path, threshold, input_size, viz)

    def estimate(self, left, right):
        right_image = [self.load_img(right)]
        left_image = [self.load_img(left)]
        probs, blinks = self.predict(left_image, right_image)

        if self.viz:
            pair_img = np.concatenate((right_image[0], left_image[0]), axis=1)
            viz_img = self.overlay_prediction_over_img(pair_img, blinks[0])
            cv2.imshow('image pair visualisation', viz_img)  
        return probs, blinks


class BlinkEstimatorFolderPair(BlinkEstimatorStandalone):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorFolderPair, self).__init__(model_path, threshold, input_size, viz)

    def estimate(self, left, right):
        left_images = []
        right_images = []
        
        for left_image_name in listdir(left):
            left_image_path = left + '/' + left_image_name
            left_images.append(self.load_img(left_image_path))

        for right_image_name in listdir(right):
            right_image_path = right + '/' + right_image_name
            right_images.append(self.load_img(right_image_path))
            
        probs, blinks = self.predict(left_images, right_images)
        if self.viz:
            for left_image, right_image, is_blinking in zip(left_images, right_images, blinks):
                pair_img = np.concatenate((right_image, left_image), axis=1)
                viz_img = self.overlay_prediction_over_img(pair_img, is_blinking)
                cv2.imshow('folder images visualisation', viz_img) 
        return probs, blinks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate blink from image or folder pair.')
    parser.add_argument('left', type=str, help='Path to an image or a directory containing images')
    parser.add_argument('right', type=str, help='Path to an image or a directory containing images')
    parser.add_argument('--model', nargs='+', type=str, default=[os.path.join(script_path, '../model_nets/Model_allsubjects1.h5')], help='List of blink estimators')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold to determine weither the prediction is positive or negative')
    parser.add_argument('--vis_blink', type=bool, default=True, help='Show the overlayed result on original image or not')             
    
    args = parser.parse_args()
    left_path = args.left
    right_path = args.right
    model_path = args.model
    threshold = args.threshold
    viz = args.vis_blink

    input_size = (96, 96)

    if os.path.isfile(left_path) and os.path.isfile(right_path):
        blink_estimator = BlinkEstimatorImagePair(model_path, threshold, input_size, viz)

    elif os.path.isdir(left_path) and os.path.isdir(right_path):
        blink_estimator = BlinkEstimatorFolderPair(model_path, threshold, input_size, viz)
    else:
        raise Exception('input inconsistency, the 2 paths must be of the same type: either file or folder')

    blinks = blink_estimator.estimate(left_path, right_path)
    print(blinks)
