"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function

import cv2
import numpy as np

from rt_gene.gaze_tools import get_normalised_eye_landmarks


class TrackedSubject(object):
    def __init__(self, box, face, landmarks):
        self.box = box
        self.face_color = face
        self.landmarks = landmarks

        self.left_eye_color = None
        self.right_eye_color = None

    def compute_distance(self, other_element):
        return np.sqrt(np.sum((self.box - other_element.box) ** 2))

    @staticmethod
    def get_eye_image_from_landmarks(subject, eye_image_size):
        eye_landmarks = get_normalised_eye_landmarks(subject.landmarks, subject.box)
        margin_ratio = 1.0
        desired_ratio = float(eye_image_size[1]) / float(eye_image_size[0]) / 2.0

        try:
            # Get the width of the eye, and compute how big the margin should be according to the width
            lefteye_width = eye_landmarks[3][0] - eye_landmarks[2][0]
            righteye_width = eye_landmarks[1][0] - eye_landmarks[0][0]

            lefteye_center_x = eye_landmarks[2][0] + lefteye_width / 2
            righteye_center_x = eye_landmarks[0][0] + righteye_width / 2
            lefteye_center_y = (eye_landmarks[2][1] + eye_landmarks[3][1]) / 2.0
            righteye_center_y = (eye_landmarks[1][1] + eye_landmarks[0][1]) / 2.0

            aligned_face, rot_matrix = GenericTracker.align_face_to_eyes(subject.face_color, right_eye_center=(righteye_center_x, righteye_center_y),
                                                                         left_eye_center=(lefteye_center_x, lefteye_center_y))

            # rotate the eye landmarks by same affine rotation to extract the correct landmarks
            ones = np.ones(shape=(len(eye_landmarks), 1))
            points_ones = np.hstack([eye_landmarks, ones])
            transformed_eye_landmarks = rot_matrix.dot(points_ones.T).T

            # recompute widths, margins and centers
            lefteye_width = transformed_eye_landmarks[3][0] - transformed_eye_landmarks[2][0]
            righteye_width = transformed_eye_landmarks[1][0] - transformed_eye_landmarks[0][0]
            lefteye_margin, righteye_margin = lefteye_width * margin_ratio, righteye_width * margin_ratio
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

            right_bb = np.zeros(4, dtype=np.int)
            right_bb[0] = transformed_eye_landmarks[0][0] - righteye_margin / 2.0
            right_bb[1] = righteye_center_y - (righteye_width + righteye_margin) * desired_ratio
            right_bb[2] = transformed_eye_landmarks[1][0] + righteye_margin / 2.0
            right_bb[3] = righteye_center_y + (righteye_width + righteye_margin) * desired_ratio

            # Extract the eye images from the aligned image
            left_eye_color = aligned_face[left_bb[1]:left_bb[3], left_bb[0]:left_bb[2], :]
            right_eye_color = aligned_face[right_bb[1]:right_bb[3], right_bb[0]:right_bb[2], :]

            # So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
            left_eye_color_resized = cv2.resize(left_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)
            right_eye_color_resized = cv2.resize(right_eye_color, eye_image_size, interpolation=cv2.INTER_CUBIC)

            return left_eye_color_resized, right_eye_color_resized, left_bb, right_bb
        except (ValueError, TypeError, cv2.error) as e:
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

    @staticmethod
    def align_face_to_eyes(face_img, right_eye_center, left_eye_center, face_width=None, face_height=None):
        # modified from https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/
        desired_left_eye = (0.35, 0.35)
        desired_face_width = face_width if face_width is not None else face_img.shape[1]
        desired_face_height = face_height if face_height is not None else face_img.shape[0]
        # compute the angle between the eye centroids
        d_y = right_eye_center[1] - left_eye_center[1]
        d_x = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(d_y, d_x)) - 180

        # compute the desired right eye x-coordinate based on the
        # desired x-coordinate of the left eye
        desired_right_eye_x = 1.0 - desired_left_eye[0]

        # determine the scale of the new resulting image by taking
        # the ratio of the distance between eyes in the *current*
        # image to the ratio of distance between eyes in the
        # *desired* image
        dist = np.sqrt((d_x ** 2) + (d_y ** 2))
        desired_dist = (desired_right_eye_x - desired_left_eye[0])
        desired_dist *= desired_face_width
        scale = desired_dist / dist

        # compute center (x, y)-coordinates (i.e., the median point)
        # between the two eyes in the input image
        eyes_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                       (left_eye_center[1] + right_eye_center[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        m = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        t_x = desired_face_width * 0.5
        t_y = desired_face_height * desired_left_eye[1]
        m[0, 2] += (t_x - eyes_center[0])
        m[1, 2] += (t_y - eyes_center[1])

        # apply the affine transformation
        (w, h) = (desired_face_width, desired_face_height)
        aligned_face = cv2.warpAffine(face_img, m, (w, h), flags=cv2.INTER_NEAREST)
        return aligned_face, m

    def update_eye_images(self, eye_image_size):
        for subject in self.get_tracked_elements().values():
            le_c, re_c, le_bb, re_bb = TrackedSubject.get_eye_image_from_landmarks(subject, eye_image_size)
            subject.left_eye_color = le_c
            subject.right_eye_color = re_c
