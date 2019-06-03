"""
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""
from __future__ import print_function


class TrackedElement(object):

    def compute_distance(self, other_element):
        raise NotImplementedError("'compute_distance' method must be overridden!")


class GenericTracker(object):

    def get_tracked_elements(self):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def clear_elements(self):
        raise NotImplementedError("'compute_distance' method must be overridden!")

    def track(self, new_elements):
        raise NotImplementedError("'compute_distance' method must be overridden!")
