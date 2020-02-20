"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function
import numpy as np
import cv2


class TrackedSubject(object):
    def __init__(self, box, face, transformed_eye_landmarks, landmarks):
        self.box = box
        self.face_color = face
        self.transformed_eye_landmarks = transformed_eye_landmarks
        self.landmarks = landmarks

        self.left_eye_color = None
        self.right_eye_color = None
        self.left_eye_bb = None
        self.right_eye_bb = None

    def compute_distance(self, other_element):
        return np.sqrt(np.sum((self.box - other_element.box) ** 2))

    @staticmethod
    def get_eye_image_from_landmarks(transformed_eye_landmarks, face_aligned_color, eye_image_size):
        margin_ratio = 1.0
        desired_ratio = float(eye_image_size[1]) / float(eye_image_size[0]) / 2.0

        try:
            # Get the width of the eye, and compute how big the margin should be according to the width
            lefteye_width = transformed_eye_landmarks[3][0] - transformed_eye_landmarks[2][0]
            righteye_width = transformed_eye_landmarks[1][0] - transformed_eye_landmarks[0][0]
            lefteye_margin, righteye_margin = lefteye_width * margin_ratio, righteye_width * margin_ratio

            # lefteye_center_x = transformed_landmarks[2][0] + lefteye_width / 2
            # righteye_center_x = transformed_landmarks[0][0] + righteye_width / 2
            lefteye_center_y = (transformed_eye_landmarks[2][1] + transformed_eye_landmarks[3][1]) / 2.0
            righteye_center_y = (transformed_eye_landmarks[1][1] + transformed_eye_landmarks[0][1]) / 2.0

            # Now compute the bounding boxes
            # The left / right x-coordinates are computed as the landmark position plus/minus the margin
            # The bottom / top y-coordinates are computed according to the desired ratio, as the width of the image is known
            left_bb = np.zeros(4, dtype=np.int)
            left_bb[0] = transformed_eye_landmarks[2][0] - lefteye_margin / 2.0
            left_bb[1] = lefteye_center_y - (lefteye_width + lefteye_margin) * desired_ratio
            left_bb[2] = transformed_eye_landmarks[3][0] + lefteye_margin / 2.0
            left_bb[3] = lefteye_center_y + (lefteye_width + lefteye_margin) * desired_ratio

            left_bb = list(map(int, left_bb))

            right_bb = np.zeros(4, dtype=np.float)
            right_bb[0] = transformed_eye_landmarks[0][0] - righteye_margin / 2.0
            right_bb[1] = righteye_center_y - (righteye_width + righteye_margin) * desired_ratio
            right_bb[2] = transformed_eye_landmarks[1][0] + righteye_margin / 2.0
            right_bb[3] = righteye_center_y + (righteye_width + righteye_margin) * desired_ratio

            right_bb = list(map(int, right_bb))

            # Extract the eye images from the aligned image
            left_eye_color = face_aligned_color[left_bb[1]:left_bb[3], left_bb[0]:left_bb[2], :]
            right_eye_color = face_aligned_color[right_bb[1]:right_bb[3], right_bb[0]:right_bb[2], :]

            # for p in transformed_landmarks:  # For debug visualization only
            #     cv2.circle(face_aligned_color, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            # So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
            left_eye_color_resized = cv2.resize(left_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)
            right_eye_color_resized = cv2.resize(right_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)

            return left_eye_color_resized, right_eye_color_resized, left_bb, right_bb
        except (ValueError, TypeError, cv2.error):
            return None, None, None, None


class GenericTracker(object):
    def __init__(self):
        self._tracked_elements = {}
        self._i = -1

    def get_tracked_elements(self):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def clear_elements(self):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def track(self, new_elements):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def get_distance_matrix(self, new_elements):
        map_index_to_id = {}  # map the matrix indexes with real unique id
        distance_matrix = np.full((len(self._tracked_elements), len(new_elements)), np.inf)
        for i, element_id in enumerate(self._tracked_elements.keys()):
            map_index_to_id[i] = element_id
            for j, new_element in enumerate(new_elements):
                # ensure new_element is of type TrackedSubject upon entry
                if not isinstance(new_element, TrackedSubject):
                    raise TypeError("Inappropriate type: {} for element whereas a TrackedSubject is expected".format(type(new_element)))
                distance_matrix[i][j] = self._tracked_elements[element_id].compute_distance(new_element)
        return distance_matrix, map_index_to_id

    def update_eye_images(self, eye_image_size):
        for subject in self.get_tracked_elements().values():
            le_c, re_c, le_bb, re_bb = subject.get_eye_image_from_landmarks(subject.transformed_eye_landmarks, subject.face_color, eye_image_size)
            subject.left_eye_color = le_c
            subject.right_eye_color = re_c
            subject.left_eye_bb = le_bb
            subject.right_eye_bb = re_bb
