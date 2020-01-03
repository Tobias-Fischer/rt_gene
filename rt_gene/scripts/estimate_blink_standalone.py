#!/usr/bin/env python

from rt_bene.blink_estimators import BlinkEstimator
from os import listdir

class BlinkEstimatorStandalone(BlinkEstimator):

    def __init__(self, model_path, threshold, input_size, viz):
        super(BlinkEstimatorStandalone, self).__init__('', 0.5, (96, 96))
        self.viz = viz
            

class BlinkEstimatorImagePair(BlinkEstimatorStandalone):

    def estimate(self, images):
        right_eyes = [images[0]]
        left_eyes = [images[1]]
        probs, blinks = self.predict(right_eyes, left_eyes)

        if self.viz:
            #TODO: viz on eye images?
            viz_img = self.overlay_prediction_over_img(right_eyes[0], blinks[0])
        return probs, blinks

class BlinkEstimatorPathPair(BlinkEstimatorStandalone):

    def estimate(self, eye_paths):
        right_eyes = [self.load_img(eye_paths[0])]
        left_eyes = [self.load_img(eye_paths[1])]
        probs, blinks = self.predict(right_eyes, left_eyes)

        if self.viz:
            #TODO: viz on eye images?
            viz_img = self.overlay_prediction_over_img(right_eyes[0], blinks[0])
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
            #TODO: viz on eye images? cv2.imshow ?
            viz_img = self.overlay_prediction_over_img(right_eyes[0], blinks[0])
        return probs, blinks

class BlinkEstimatorVideo(BlinkEstimatorStandalone):
    # TODO
    def estimate(self, video_path):
        pass



if __name__ == '__main__':
    #TODO arg parser
    viz = True
    model_path = ''
    threshold = 0.5
    input_size = (96, 96)
    mode = 'pair'
    right_eye_path = ''
    left_eye_path = ''
    folder_path = ''
    video_path = ''

    if mode == 'path_pair':
        blink_estimator = BlinkEstimatorPathPair(model_path, threshold, input_size, viz)
        input_path = [right_eye_path, left_eye_path]
    elif mode == 'folder':
        blink_estimator = BlinkEstimatorFolder(model_path, threshold, input_size, viz)
        input_path = folder_path
    elif mode == 'video':
        blink_estimator = BlinkEstimatorVideo(model_path, threshold, input_size, viz)
        input_path = video_path

    blinks = blink_estimator.estimate(input_path)