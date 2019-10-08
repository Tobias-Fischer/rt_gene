"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

import rospy
from scipy import optimize
import scipy
from .tracker_generic import GenericTracker


class SequentialTracker(GenericTracker):
    def __init__(self):
        super(SequentialTracker, self).__init__()
        rospy.logwarn("** SequentialTracker is no longer supported, please use the FaceEncodingTracker instead")

    ''' --------------------------------------------------------------------'''
    ''' PRIVATE METHODS '''

    def __add_new_element(self, element):
        new_id = self._generate_unique_id()
        self._tracked_elements[new_id] = element
        return new_id

    def __update_element(self, element_id, element):
        self._tracked_elements[element_id] = element

    ''' --------------------------------------------------------------------'''
    ''' PROTECTED METHODS '''

    # (can be overridden if necessary)
    def _generate_unique_id(self):
        self._i += 1
        return str(self._i)

    ''' --------------------------------------------------------------------'''
    ''' PUBLIC METHODS '''

    def get_tracked_elements(self):
        return self._tracked_elements

    def clear_elements(self):
        self._tracked_elements.clear()

    def track(self, new_elements):
        # if no elements yet, just add all the new ones
        if len(self._tracked_elements) == 0:
            [self.__add_new_element(e) for e in new_elements]
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
                match_idx = col[row_list.index(j)]
                # if the new element matches with existing old one
                matched_element_id = map_index_to_id[match_idx]
                self.__update_element(matched_element_id, new_element)
                _new_idx = matched_element_id

            else:
                # if the new element is not matching
                _new_idx = self.__add_new_element(new_element)
            updated_tracked_element_ids.append(_new_idx)

        # delete all the non-updated elements
        elements_to_delete = list(set(current_tracked_element_ids) - set(updated_tracked_element_ids))
        for i in elements_to_delete:
            del self._tracked_elements[i]
