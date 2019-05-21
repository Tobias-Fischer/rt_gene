"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
from __future__ import print_function
import numpy as np
import scipy.optimize


class TrackedElement(object):

    def encode(self):
        raise NotImplementedError("'encode' method must be overridden!")

    def compute_distance(self, other_element):
        raise NotImplementedError("'compute_distance' method must be overridden!")


class GenericTracker(object):
    def __init__(self):
        self.__tracked_elements = {}
        self.__removed_elements = {}
        self.__i = -1
        self.__threshold = 0.6

    ''' --------------------------------------------------------------------'''
    ''' PRIVATE METHODS '''

    def __add_new_element(self, element):
        # encode the new array
        encoding = np.array(element.encode())
        found_previous = False

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

    ''' --------------------------------------------------------------------'''
    ''' PROTECTED METHODS '''

    # (can be overridden if necessary)
    def _generate_new_id(self):
        self.__i += 1
        return self.__i

    ''' --------------------------------------------------------------------'''
    ''' PUBLIC METHODS '''

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
