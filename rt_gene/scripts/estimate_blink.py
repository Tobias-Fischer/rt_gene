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
from rt_gene.msg import MSG_BlinkList, MSG_Blink
from rt_gene.subject_ros_bridge import SubjectListBridge

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

import numpy as np
import collections
from tqdm import tqdm


class BlinkEstimatorROS(object):
    def __init__(self, device_id_blink, model_files, threshold):
        self.cv_bridge = CvBridge()
        self.bridge = SubjectListBridge()
        self.viz = rospy.get_param("~viz", True)

        blink_backend = rospy.get_param("~blink_backend", default="pytorch")
        model_type = rospy.get_param("~model_type", default="resnet18")

        if blink_backend == "tensorflow":
            from rt_bene.estimate_blink_tensorflow import BlinkEstimatorTensorflow
            self._blink_estimator = BlinkEstimatorTensorflow(device_id_blink, model_files, model_type, threshold)
        elif blink_backend == "pytorch":
            from rt_bene.estimate_blink_pytorch import BlinkEstimatorPytorch
            self._blink_estimator = BlinkEstimatorPytorch(device_id_blink, model_files, model_type, threshold)
        else:
            raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")

        self._last_time = rospy.Time().now()
        self._freq_deque = collections.deque(maxlen=30)  # average frequency statistic over roughly one second
        self._latency_deque = collections.deque(maxlen=30)

        self.blink_publisher = rospy.Publisher("/subjects/blink", MSG_BlinkList, queue_size=3)
        if self.viz:
            self.viz_pub = rospy.Publisher(rospy.get_param("~viz_topic", "/subjects/blink_images"), Image, queue_size=3)

        self.sub = rospy.Subscriber("/subjects/images", MSG_SubjectImagesList, self.callback, queue_size=1,
                                    buff_size=2 ** 24)

    def callback(self, msg):
        subjects = self.bridge.msg_to_images(msg)
        left_eyes = []
        right_eyes = []

        for subject in subjects.values():
            _left, _right = self._blink_estimator.inputs_from_images(subject.left, subject.right)
            left_eyes.append(_left)
            right_eyes.append(_right)

        if len(left_eyes) == 0:
            return

        probs = self._blink_estimator.predict(left_eyes, right_eyes)

        self.publish_msg(msg.header, subjects, probs)

        if self.viz:
            blink_image_list = []
            for subject, p in zip(subjects.values(), probs):
                resized_face = cv2.resize(subject.face, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                blink_image_list.append(self._blink_estimator.overlay_prediction_over_img(resized_face, p))

            if len(blink_image_list) > 0:
                blink_viz_img = self.cv_bridge.cv2_to_imgmsg(np.hstack(blink_image_list), encoding="bgr8")
                blink_viz_img.header.stamp = msg.header.stamp
                self.viz_pub.publish(blink_viz_img)

        _now = rospy.Time().now()
        timestamp = msg.header.stamp

        _freq = 1.0 / (_now - self._last_time).to_sec()
        self._freq_deque.append(_freq)
        self._latency_deque.append(_now.to_sec() - timestamp.to_sec())
        self._last_time = _now
        tqdm.write(
            '\033[2K\033[1;32mTime now: {:.2f} message color: {:.2f} latency: {:.2f}s for {} subject(s) {:.0f}Hz\033[0m'.format(
                (_now.to_sec()), timestamp.to_sec(), np.mean(self._latency_deque), len(subjects),
                np.mean(self._freq_deque)), end="\r")

    def publish_msg(self, header, subjects, probabilities):
        blink_msg_list = MSG_BlinkList()
        blink_msg_list.header = header
        for subject_id, p in zip(subjects.keys(), probabilities):
            blink_msg = MSG_Blink()
            blink_msg.subject_id = str(subject_id)
            blink_msg.blink = bool(p >= self._blink_estimator.threshold)
            blink_msg.probability = p
            blink_msg_list.subjects.append(blink_msg)

        self.blink_publisher.publish(blink_msg_list)


if __name__ == "__main__":
    try:
        rospy.init_node("blink_estimator")
        blink_detector = BlinkEstimatorROS(device_id_blink=rospy.get_param("~device_id_blinkestimation", "/gpu:0"),
                                           model_files=[os.path.join(rospkg.RosPack().get_path("rt_gene"), model_file)
                                                        for model_file in rospy.get_param("~model_files")],
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
