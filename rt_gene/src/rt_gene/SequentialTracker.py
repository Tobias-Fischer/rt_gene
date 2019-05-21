"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
import numpy as np
import scipy
from GenericTracker import TrackedElement, GenericTracker


class SequentialTracker(GenericTracker):
    def __init__(self):
        self.__tracked_elements = {}
        self.__i = -1

    ''' --------------------------------------------------------------------'''
    ''' PRIVATE METHODS '''

    def __add_new_element(self, element):
        self.__tracked_elements[self._generate_unique_id()] = element

    def __update_element(self, element_id, element):
        self.__tracked_elements[element_id] = element

    ''' --------------------------------------------------------------------'''
    ''' PROTECTED METHODS '''

    # (can be overridden if necessary)
    def _generate_unique_id(self):
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
                match_idx = col[row.tolist().index(j)]
                # if the new element matches with existing old one
                matched_element_id = map_index_to_id[match_idx]
                self.__update_element(matched_element_id, new_element)
                updated_tracked_element_ids.append(matched_element_id)
            except ValueError:
                # if the new element is not matching
                self.__add_new_element(new_element)

        # delete all the non-updated elements
        elements_to_delete = list(set(current_tracked_element_ids) - set(updated_tracked_element_ids))
        for i in elements_to_delete:
            del self.__tracked_elements[i]
