"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
import numpy as np
import scipy
import uuid


def isinstance_tracked_element(element):
    if not isinstance(element, TrackedElement):
        raise ValueError('Inappropriate type: {} for element whereas a TrackedElement is expected'.format(type(element)))


class TrackedElement(object):
    def compute_distance(self, other_element):
        raise NotImplementedError("'compute_distance' method must be overridden!")


class GenericTracker(object):
    def __init__(self):
        self.__tracked_elements = {}

    ''' --------------------------------------------------------------------'''
    ''' PRIVATE METHODS '''

    def __add_new_element(self, element):
        isinstance_tracked_element(element)
        self.__tracked_elements[self._generate_unique_id()] = element

    def __update_element(self, element_id, element):
        isinstance_tracked_element(element)
        self.__tracked_elements[element_id] = element

    def __clear_elements(self):
        self.__tracked_elements.clear()

    ''' --------------------------------------------------------------------'''
    ''' PROTECTED METHODS '''

    # (can be overridden if necessary)
    def _generate_unique_id(self):
        return uuid.uuid4().hex

    ''' --------------------------------------------------------------------'''
    ''' PUBLIC METHODS '''

    def get_tracked_elements(self):
        return self.__tracked_elements

    def track(self, new_elements):
        # if no new elements, remove old elements
        if not new_elements:
            self.__clear_elements()
            return

        # if no elements yet, just add all the new ones
        if not self.__tracked_elements:
            [self.__add_new_element(e) for e in new_elements]
            return

        current_tracked_element_ids = self.__tracked_elements.keys()
        updated_tracked_element_ids = []
        map_index_to_id = {} # map the matrix indexes with real unique id

        distance_matrix = np.ones((len(self.__tracked_elements), len(new_elements)))
        for i, element_id in enumerate(self.__tracked_elements.keys()):
            map_index_to_id[i] = element_id
            for j, new_element in enumerate(new_elements):
                distance_matrix[i][j] = self.__tracked_elements[element_id].compute_distance(new_element)

        # get best matching pairs with Hungarian Algorithm
        col, row = scipy.optimize.linear_sum_assignment(distance_matrix)

        # assign each new element to existing one or store it as new
        for j, new_element in enumerate(new_elements):
            try:
                # if the new element matches with existing old one
                matched_element_id = map_index_to_id[col[row.tolist().index(j)]]
                self.__update_element(matched_element_id, new_element)
                updated_tracked_element_ids.append(matched_element_id)
            except ValueError:
                # if the new element is not matching
                self.__add_new_element(new_element)

        # delete all the non-updated elements
        elements_to_delete = list(set(current_tracked_element_ids) - set(updated_tracked_element_ids))
        for i in elements_to_delete:
            del self.__tracked_elements[i]   
