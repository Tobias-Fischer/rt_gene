#!/usr/bin/env python

"""
CNN for blink estimation
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Tobias Fischer (t.fischer@imperial.ac.uk)
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import os
import rospy
import rospkg

from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.subject_ros_bridge import SubjectListBridge
from rt_bene.estimate_blink_base import BlinkEstimatorBase

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import numpy as np


class BlinkEstimatorNode(BlinkEstimatorBase):
    def __init__(self, device_id_blink, model_files, threshold):
        super(BlinkEstimatorNode, self).__init__(device_id_blink, model_files, threshold, (96, 96))
        self.sub = rospy.Subscriber("/subjects/images", MSG_SubjectImagesList, self.callback, queue_size=1, buff_size=2**24)
        self.cv_bridge = CvBridge()
        self.bridge = SubjectListBridge()
        self.viz = rospy.get_param("~viz", True)

        if self.viz:
            self.viz_pub = rospy.Publisher(rospy.get_param("~viz_topic", "/subjects/blinks"), Image, queue_size=1)

    def callback(self, msg):
        subjects = self.bridge.msg_to_images(msg)
        right_eyes = []
        left_eyes = []

        for subject in subjects.values():
            # right_eyes.append(cv2.flip(cv2.cvtColor(self.resize_img(subject.right), cv2.COLOR_RGB2BGR), 1))
            # left_eyes.append(cv2.cvtColor(self.resize_img(subject.left), cv2.COLOR_RGB2BGR))
            right_eyes.append(cv2.flip(self.resize_img(subject.right), 1))
            left_eyes.append(self.resize_img(subject.left))

        if len(left_eyes) == 0:
            return

        probs, _ = self.predict(right_eyes, left_eyes)

        if self.viz:
            blink_image_list = []
            for subject, p in zip(subjects.values(), probs):
                resized_face = cv2.resize(subject.face, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                blink_image_list.append(self.overlay_prediction_over_img(resized_face, p))

            if len(blink_image_list) > 0:
                blink_viz_img = self.cv_bridge.cv2_to_imgmsg(np.hstack(blink_image_list))
                blink_viz_img.header.stamp = msg.header.stamp
                self.viz_pub.publish(blink_viz_img)


if __name__ == "__main__":
    try:
        rospy.init_node("blink_estimator")
        blink_detector = BlinkEstimatorNode(device_id_blink=rospy.get_param("~device_id_blinkestimation", "/gpu:0"),
                                            model_files=[os.path.join(rospkg.RosPack().get_path("rt_gene"), model_file) for model_file in rospy.get_param("~model_files")],
                                            threshold=rospy.get_param("~threshold", 0.5))
        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        print("See ya")
    except rospy.ROSException as e:
        if str(e) == "publish() to a closed topic":
            print("See ya")
        else:
            raise e
    except KeyboardInterrupt:
        print("Shutting down")
