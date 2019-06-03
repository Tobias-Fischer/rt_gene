"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
from __future__ import print_function
import numpy as np
import scipy.optimize
from GenericTracker import GenericTracker, TrackedElement
import cv2
import dlib
import rospkg


class FaceEncodingTracker(GenericTracker):

    FACE_ENCODER = dlib.face_recognition_model_v1(
        rospkg.RosPack().get_path('rt_gene') + '/model_nets/dlib_face_recognition_resnet_model_v1.dat')

    def __init__(self):
        self.__tracked_elements = {}
        self.__removed_elements = {}
        self.__i = -1
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
        found_previous = False

        encoding = np.array(element.encode(self.__encode_subject(element)))
        # check to see if we've seen it before
        for i, previous_encoding in enumerate(self.__removed_elements.keys()):
            previous_encoding = np.fromstring(previous_encoding[1:-1], dtype=np.float, sep=",")
            distance = np.linalg.norm(previous_encoding - encoding, axis=0)

            # the new element and the previous encoding are the same person
            if distance < self.__threshold:
                previous_id = self.__removed_elements.values()[i]
                self.__tracked_elements[previous_id] = element
                found_previous = True
                break

        if not found_previous:
            self.__tracked_elements[self._generate_new_id()] = element

    def __update_element(self, element_id, element):
        self.__tracked_elements[element_id] = element

    # (can be overridden if necessary)
    def _generate_new_id(self):
        self.__i += 1
        return self.__i

    def get_tracked_elements(self):
        return self.__tracked_elements

    def clear_elements(self):
        self.__tracked_elements.clear()

    def track(self, new_elements):
        # if no new elements, remove old elements
        if not new_elements:
            self.clear_elements()
            return

        # if no elements yet, just add all the new ones
        if not self.__tracked_elements:
            [self.__add_new_element(e) for e in new_elements]
            return

        current_tracked_element_ids = self.__tracked_elements.keys()
        updated_tracked_element_ids = []
        map_index_to_id = {}  # map the matrix indexes with real unique id

        distance_matrix = np.ones((len(self.__tracked_elements), len(new_elements)))
        for i, element_id in enumerate(self.__tracked_elements.keys()):
            map_index_to_id[i] = element_id
            for j, new_element in enumerate(new_elements):
                # ensure new_element is of type TrackedElement upon entry
                if not isinstance(new_element, TrackedElement):
                    raise TypeError("Inappropriate type: {} for element whereas a TrackedElement is expected".format(
                        type(new_element)))
                distance_matrix[i][j] = self.__tracked_elements[element_id].compute_distance(new_element)

        # get best matching pairs with Hungarian Algorithm
        col, row = scipy.optimize.linear_sum_assignment(distance_matrix)

        # assign each new element to existing one or store it as new
        for j, new_element in enumerate(new_elements):
            try:
                # find the index of the column matching
                match_idx = col[np.min(np.nonzero(row == j))]
                # if the new element matches with existing old one
                matched_element_id = map_index_to_id[match_idx]
                self.__update_element(matched_element_id, new_element)
                updated_tracked_element_ids.append(matched_element_id)
            except ValueError:
                # if the new element is not matching
                self.__add_new_element(new_element)

        # store non-tracked elements in-case they reappear
        elements_to_delete = list(set(current_tracked_element_ids) - set(updated_tracked_element_ids))
        for i in elements_to_delete:
            _element = self.__tracked_elements[i]  # the subject itself
            encoding = np.array2string(np.array(_element.encode()), formatter={'float_kind': lambda x: "%.5f" % x},
                                       separator=",")
            # store the encoding and it's respective key
            self.__removed_elements[encoding] = i

            # don't track it anymore
            del self.__tracked_elements[i]
