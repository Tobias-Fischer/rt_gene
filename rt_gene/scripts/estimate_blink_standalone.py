#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

from os import listdir
from rt_bene.estimate_blink_base import BlinkEstimatorBase
import sys
import os
import argparse

import cv2

script_path = os.path.dirname(os.path.realpath(__file__))

class BlinkEstimatorStandalone(BlinkEstimatorBase):

    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorStandalone, self).__init__("gpu", model_path, threshold, input_size)
        self.viz = viz
            

class BlinkEstimatorImagePair(BlinkEstimatorStandalone):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorImagePair, self).__init__(model_path, threshold, input_size, viz)

    def estimate(self, images):
        right_eyes = [images[0]]
        left_eyes = [images[1]]
        probs, blinks = self.predict(right_eyes, left_eyes)

        if self.viz:
            #TODO: viz on eye images?
            viz_img = self.overlay_prediction_over_img(right_eyes[0], blinks[0])
            cv2.imshow('image pair vis', viz_img)  
        return probs, blinks

class BlinkEstimatorPathPair(BlinkEstimatorStandalone):
    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorPathPair, self).__init__(model_path, threshold, input_size, viz)
        
    def estimate(self, eye_paths):
        right_eyes = [self.load_img(eye_paths[0])]
        left_eyes = [self.load_img(eye_paths[1])]
        probs, blinks = self.predict(right_eyes, left_eyes)

        if self.viz:
            viz_img = self.overlay_prediction_over_img(right_eyes[0], blinks[0])
            cv2.imshow('image pair vis', viz_img) 
        return probs, blinks

class BlinkEstimatorFolder(BlinkEstimatorStandalone):

    def estimate(self, folder_path):
        right_eyes = []
        left_eyes = []
        for file_name in [f for f in listdir(folder_path) if 'right' not in f]:
            right_eye_path = folder_path + '/' + file_name.replace('left', 'right')
            left_eye_path = folder_path + '/' + file_name
            right_eyes.append(self.load_img(right_eye_path))
            left_eyes.append(self.load_img(left_eye_path))
        probs, blinks = self.predict(right_eyes, left_eyes)
        if self.viz:
            viz_img = self.overlay_prediction_over_img(right_eyes[0], blinks[0])
            cv2.imshow('folder images vis', viz_img) 
        return probs, blinks


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate blink from images, folder or even sequence.')
    parser.add_argument('mode', type=str, help='the input mode', choices=['image-pair', 'folder'])
    parser.add_argument('left', type=str, help='Path to an image or a directory containing images')
    parser.add_argument('right', type=str, help='Path to an image or a directory containing images')
    parser.add_argument('--model', nargs='+', type=str, default=[os.path.join(script_path, '../model_nets/Model_allsubjects1.h5')], help='List of blink estimators')
    parser.add_argument('--threshold', type=float, default=0.5, help='')
    parser.add_argument('--vis_blink', type=bool, default=True, help='')             
    args = parser.parse_args()
    
    viz = args.vis_blink
    model_path = args.model
    threshold = args.threshold
    input_size = (96, 96)
    mode = args.mode
    right_eye_path = args.right
    left_eye_path = args.left
    folder_path = ''
    video_path = ''

    if mode == 'image-pair':
        blink_estimator = BlinkEstimatorPathPair(model_path, threshold, input_size, viz)
        input_path = [right_eye_path, left_eye_path]
    elif mode == 'folder':
        blink_estimator = BlinkEstimatorFolder(model_path, threshold, input_size, viz)
        input_path = folder_path

    blinks = blink_estimator.estimate(input_path)
    print(blinks)
