"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
from __future__ import print_function
import numpy as np


class TrackedElement(object):
    def compute_distance(self, other_element):
        raise NotImplementedError("'compute_distance' method must be overridden!")


class GenericTracker(object):
    def __init__(self):
        self.__tracked_elements = {}
        self.__i = -1

    def get_tracked_elements(self):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def clear_elements(self):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def track(self, new_elements):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def get_distance_matrix(self, new_elements):
        map_index_to_id = {}  # map the matrix indexes with real unique id
        distance_matrix = np.full((len(self.__tracked_elements), len(new_elements)), np.inf)
        for i, element_id in enumerate(self.__tracked_elements.keys()):
            map_index_to_id[i] = element_id
            for j, new_element in enumerate(new_elements):
                # ensure new_element is of type TrackedElement upon entry
                if not isinstance(new_element, TrackedElement):
                    raise TypeError("Inappropriate type: {} for element whereas a TrackedElement is expected".format(type(new_element)))
                distance_matrix[i][j] = self.__tracked_elements[element_id].compute_distance(new_element)
        return distance_matrix, map_index_to_id
