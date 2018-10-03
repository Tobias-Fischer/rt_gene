#!/usr/bin/env python

"""
A wrapper around the scripts in src/rt_gene/extract_landmarks_*.py
@Tobias Fischer (t.fischer@imperial.ac.uk)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import rospy
# noinspection PyUnresolvedReferences
import rt_gene.extract_landmarks_new

if __name__ == '__main__':
    try:
        rospy.init_node('extract_landmarks')

        landmark_extractor = rt_gene.extract_landmarks_new.LandmarkNew()

        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        print("See ya")
    except KeyboardInterrupt:
        print("Shutting down")

