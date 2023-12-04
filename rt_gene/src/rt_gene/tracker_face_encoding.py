"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function

import cv2
import dlib
import numpy as np
import rospkg
import rospy
import scipy.optimize

from rt_gene.gaze_tools import get_normalised_eye_landmarks
from .tracker_generic import GenericTracker


class FaceEncodingTracker(GenericTracker):
    FACE_ENCODER = dlib.face_recognition_model_v1(
        rospkg.RosPack().get_path('rt_gene') + '/model_nets/dlib_face_recognition_resnet_model_v1.dat')

    def __init__(self):
        super(FaceEncodingTracker, self).__init__()
        self.__encoding_list = {}
        self.__threshold = float(rospy.get_param("~face_encoding_threshold", default=0.6))

    def __encode_subject(self, tracked_element):
        # get the face_color and face_chip it using the transformed_eye_landmarks
        eye_landmarks = get_normalised_eye_landmarks(tracked_element.landmarks, tracked_element.box)
        # Get the width of the eye, and compute how big the margin should be according to the width
        lefteye_width = eye_landmarks[3][0] - eye_landmarks[2][0]
        righteye_width = eye_landmarks[1][0] - eye_landmarks[0][0]

        lefteye_center_x = eye_landmarks[2][0] + lefteye_width / 2
        righteye_center_x = eye_landmarks[0][0] + righteye_width / 2
        lefteye_center_y = (eye_landmarks[2][1] + eye_landmarks[3][1]) / 2.0
        righteye_center_y = (eye_landmarks[1][1] + eye_landmarks[0][1]) / 2.0
        aligned_face, rot_matrix = GenericTracker.align_face_to_eyes(tracked_element.face_color,
                                                                     right_eye_center=(righteye_center_x, righteye_center_y),
                                                                     left_eye_center=(lefteye_center_x, lefteye_center_y),
                                                                     face_width=150,
                                                                     face_height=150)
        encoding = self.FACE_ENCODER.compute_face_descriptor(aligned_face)
        return encoding

    def __add_new_element(self, element):
        # encode the new array
        found_id = None

        encoding = np.array(self.__encode_subject(element))
        # check to see if we've seen it before
        list_to_check = list(set(self.__encoding_list.keys()) - set(self._tracked_elements.keys()))

        for untracked_encoding_id in list_to_check:
            previous_encoding = self.__encoding_list[untracked_encoding_id]
            previous_encoding = np.fromstring(previous_encoding[1:-1], dtype=float, sep=",")
            distance = np.linalg.norm(previous_encoding - encoding, axis=0)

            # the new element and the previous encoding are the same person
            if distance < self.__threshold:
                self._tracked_elements[untracked_encoding_id] = element
                found_id = untracked_encoding_id
                break

        if found_id is None:
            found_id = self._generate_new_id()
            self._tracked_elements[found_id] = element

            self.__encoding_list[found_id] = np.array2string(encoding, formatter={'float_kind': lambda x: "{:.5f}".format(x)}, separator=",")

        return found_id

    def __update_element(self, element_id, element):
        self._tracked_elements[element_id] = element

    # (can be overridden if necessary)
    def _generate_new_id(self):
        self._i += 1
        return str(self._i)

    def get_tracked_elements(self):
        return self._tracked_elements

    def clear_elements(self):
        self._tracked_elements.clear()

    def track(self, new_elements):
        # if no elements yet, just add all the new ones
        if not self._tracked_elements:
            for e in new_elements:
                try:
                    self.__add_new_element(e)
                except cv2.error:
                    pass
            return

        current_tracked_element_ids = self._tracked_elements.keys()
        updated_tracked_element_ids = []
        distance_matrix, map_index_to_id = self.get_distance_matrix(new_elements)

        # get best matching pairs with Hungarian Algorithm
        col, row = scipy.optimize.linear_sum_assignment(distance_matrix)

        # assign each new element to existing one or store it as new
        for j, new_element in enumerate(new_elements):
            row_list = row.tolist()
            if j in row_list:
                # find the index of the column matching
                row_idx = row_list.index(j)

                match_idx = col[row_idx]
                # if the new element matches with existing old one
                _new_idx = map_index_to_id[match_idx]
                self.__update_element(_new_idx, new_element)
                updated_tracked_element_ids.append(_new_idx)
            else:
                try:
                    _new_idx = self.__add_new_element(new_element)
                    updated_tracked_element_ids.append(_new_idx)
                except cv2.error:
                    pass

        # store non-tracked elements in-case they reappear
        elements_to_delete = list(set(current_tracked_element_ids) - set(updated_tracked_element_ids))
        for i in elements_to_delete:
            # don't track it anymore
            del self._tracked_elements[i]
