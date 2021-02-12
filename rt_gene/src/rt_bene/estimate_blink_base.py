#! /usr/bin/env python

import cv2

BLINK_COLOR = (0, 0, 255)
NO_BLINK_COLOR = (0, 255, 0)


class BlinkEstimatorBase(object):
    def __init__(self, device_id, threshold):
        self.device_id = device_id
        self.threshold = threshold

    def inputs_from_images(self, left, right):
        pass

    def predict(self, left_eyes, right_eyes):
        pass

    def overlay_prediction_over_img(self, img, p, border_size=5):
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        if p > self.threshold:
            cv2.rectangle(img_copy, (0, 0), (w, h), BLINK_COLOR, border_size)
        else:
            cv2.rectangle(img_copy, (0, 0), (w, h), NO_BLINK_COLOR, border_size)
        return img_copy
