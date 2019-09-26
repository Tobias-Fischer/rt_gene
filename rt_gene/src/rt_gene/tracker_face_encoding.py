"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
from __future__ import print_function
import numpy as np
import scipy.optimize
from .tracker_generic import GenericTracker
import cv2
import dlib
import rospkg


class FaceEncodingTracker(GenericTracker):

    FACE_ENCODER = dlib.face_recognition_model_v1(
        rospkg.RosPack().get_path('rt_gene') + '/model_nets/dlib_face_recognition_resnet_model_v1.dat')

    def __init__(self):
        super(FaceEncodingTracker, self).__init__()
        self.__encoding_list = {}
        self.__threshold = 0.6

    @staticmethod
    def __align_tracked_subject(tracked_subject, desired_left_eye=(0.3, 0.3), desired_face_width=150, desired_face_height=150):
        # extract the left and right eye (x, y)-coordinates
        right_eye_pts = np.array([tracked_subject.transformed_landmarks[0], tracked_subject.transformed_landmarks[1]])
        left_eye_pts = np.array([tracked_subject.transformed_landmarks[2], tracked_subject.transformed_landmarks[3]])

        # compute the center of mass for each eye
        left_eye_centre = left_eye_pts.mean(axis=0).astype("int")
        right_eye_centre = right_eye_pts.mean(axis=0).astype("int")

        # compute the angle between the eye centroids
        d_y = right_eye_centre[1] - left_eye_centre[1]
        d_x = right_eye_centre[0] - left_eye_centre[0]
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
        eyes_center = ((left_eye_centre[0] + right_eye_centre[0]) // 2,
                       (left_eye_centre[1] + right_eye_centre[1]) // 2)

        # grab the rotation matrix for rotating and scaling the face
        rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale)

        # update the translation component of the matrix
        t_x = desired_face_width * 0.5
        t_y = desired_face_height * desired_left_eye[1]
        rotation_matrix[0, 2] += (t_x - eyes_center[0])
        rotation_matrix[1, 2] += (t_y - eyes_center[1])

        # apply the affine transformation
        output = cv2.warpAffine(tracked_subject.face_color, rotation_matrix, (desired_face_width, desired_face_height),
                                flags=cv2.INTER_CUBIC)

        # return the aligned face
        return output

    def __encode_subject(self, tracked_element):
        # get the face_color and face_chip it using the transformed_landmarks
        face_chip = self.__align_tracked_subject(tracked_element)
        encoding = self.FACE_ENCODER.compute_face_descriptor(face_chip)
        return encoding

    def __add_new_element(self, element):
        # encode the new array
        found_id = None

        encoding = np.array(self.__encode_subject(element))
        # check to see if we've seen it before
        list_to_check = list(set(self.__encoding_list.keys()) - set(self._tracked_elements.keys()))

        for untracked_encoding_id in list_to_check:
            previous_encoding = self.__encoding_list[untracked_encoding_id]
            previous_encoding = np.fromstring(previous_encoding[1:-1], dtype=np.float, sep=",")
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
