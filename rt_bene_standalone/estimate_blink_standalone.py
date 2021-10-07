#!/usr/bin/env python

from __future__ import print_function, division, absolute_import

import argparse
import os
import time
from os import listdir

import cv2
import numpy as np

script_path = os.path.dirname(os.path.realpath(__file__))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BlinkEstimatorFolderPair(object):
    def __init__(self, blink_estimator, viz):
        self.blink_estimator = blink_estimator
        self.viz = viz

    def estimate(self, left_folder_path, right_folder_path):
        left_image_paths, right_image_paths = [], []
        left_images, right_images = [], []

        for left_image_name in sorted(listdir(left_folder_path)):
            left_image_path = left_folder_path + '/' + left_image_name
            left_image_paths.append(left_image_path)
            left_images.append(cv2.imread(left_image_path, cv2.IMREAD_COLOR))

        for right_image_name in sorted(listdir(right_folder_path)):
            right_image_path = right_folder_path + '/' + right_image_name
            right_image_paths.append(right_image_path)
            right_images.append(cv2.imread(right_image_path, cv2.IMREAD_COLOR))

        l_images_input, r_images_input = [], []
        for l_img, r_img in zip(left_images, right_images):
            l_img_input, r_img_input = self.blink_estimator.inputs_from_images(l_img, r_img)
            l_images_input.append(l_img_input)
            r_images_input.append(r_img_input)

        start_time = time.time()
        probs = self.blink_estimator.predict(l_images_input, r_images_input)
        blinks = probs >= self.blink_estimator.threshold
        print(
            "Estimated blink for {} eye-image pairs, Time: {:.5f}s".format(len(left_images), time.time() - start_time))
        if self.viz:
            for left_image, right_image, is_blinking in zip(left_images, right_images, blinks):
                pair_img = np.concatenate((right_image, left_image), axis=1)
                viz_img = self.blink_estimator.overlay_prediction_over_img(pair_img, is_blinking)
                cv2.imshow('folder images visualisation', viz_img)
                cv2.waitKey(0)
        for left_image, right_image, p, is_blinking in zip(left_image_paths, right_image_paths, probs, blinks):
            print("Blink: %s (p=%.3f) for image pair: %20s %20s" % ("Yes" if is_blinking else "No ", p,
                                                                    os.path.basename(left_image),
                                                                    os.path.basename(right_image)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate blink from image or folder pair.')
    parser.add_argument('--left', type=str, help='Path to a left eye image or a directory containing left eye images',
                        default=os.path.join(script_path, './samples_blink/left/'))
    parser.add_argument('--right', type=str,
                        help='Path to a right eye image or a directory containing images right eye images',
                        default=os.path.join(script_path, './samples_blink/right/'))
    parser.add_argument('--model', nargs='+', type=str,
                        default=[os.path.abspath(os.path.join(script_path, '../rt_gene/model_nets/blink_model_pytorch_vgg16_allsubjects1.model'))],
                        help='List of blink estimators')
    parser.add_argument('--model_type', type=str, default="vgg16")
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold to determine weither the prediction is positive or negative')
    parser.add_argument('--device_id', type=str, default="cuda")
    parser.add_argument('--blink_backend', type=str, choices=['tensorflow', 'pytorch'], default='pytorch')
    parser.add_argument('--vis_blink', type=str2bool, nargs='?', default=True,
                        help='Show the overlayed result on original image or not')

    args = parser.parse_args()
    left_path = args.left
    right_path = args.right

    if args.blink_backend == "tensorflow":
        from rt_bene.estimate_blink_tensorflow import BlinkEstimatorTensorflow
        blink_estimator = BlinkEstimatorTensorflow(device_id_blink=args.device_id, threshold=0.425, model_files=args.model, model_type=args.model_type)
    elif args.blink_backend == "pytorch":
        from rt_bene.estimate_blink_pytorch import BlinkEstimatorPytorch
        blink_estimator = BlinkEstimatorPytorch(device_id_blink=args.device_id, threshold=0.425, model_files=args.model, model_type=args.model_type)
    else:
        raise Exception("Unknown backend")

    if os.path.isdir(left_path) and os.path.isdir(right_path):
        blink_folder = BlinkEstimatorFolderPair(blink_estimator, viz=args.vis_blink)
        blink_folder.estimate(left_path, right_path)
        if args.vis_blink:
            cv2.destroyAllWindows()
    else:
        raise Exception('Folders not found: Check that ' + left_path + ' and ' + right_path + ' exist')


