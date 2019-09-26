"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
from __future__ import print_function
import numpy as np


class TrackedSubject(object):
    def __init__(self, box, face, landmarks, marks):
        super(TrackedSubject, self).__init__()
        self.box = box
        self.face_color = face
        self.transformed_landmarks = landmarks
        self.marks = marks

    # override method
    def compute_distance(self, other_element):
        return np.sqrt(np.sum((self.box - other_element.box) ** 2))


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
