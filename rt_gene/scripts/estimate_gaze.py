#!/usr/bin/env python

"""
Convolutional Neural Network (CNN) for eye gaze estimation
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Hyung Jin Chang (hj.chang@imperial.ac.uk)
@Kevin Cortacero <cortacero.k31130@gmail.com>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import os
import numpy as np
from tqdm import tqdm

import rospkg
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

# noinspection PyUnresolvedReferences
import rt_gene.gaze_tools as gaze_tools
from tf import TransformBroadcaster, TransformListener
import tf.transformations
import collections

from rt_gene.subject_ros_bridge import SubjectListBridge
from rt_gene.msg import MSG_SubjectImagesList

from rt_gene.estimate_gaze_base import GazeEstimatorBase


class GazeEstimatorROS(GazeEstimatorBase):
    def __init__(self, device_id_gaze, model_files):
        super(GazeEstimatorROS, self).__init__(device_id_gaze, model_files)
        self.bridge = CvBridge()
        self.subjects_bridge = SubjectListBridge()

        self.tf_broadcaster = TransformBroadcaster()
        self.tf_listener = TransformListener()

        self.tf_prefix = rospy.get_param("~tf_prefix", "gaze")
        self.headpose_frame = self.tf_prefix + "/head_pose_estimated"
        self.rgb_frame_id_ros = rospy.get_param("~rgb_frame_id_ros", "/kinect2_nonrotated_link")

        self.image_subscriber = rospy.Subscriber('/subjects/images', MSG_SubjectImagesList, self.image_callback, queue_size=1, buff_size=10000000)
        self.subjects_gaze_img = rospy.Publisher('/subjects/gazeimages', Image, queue_size=3)

        self.time_last = rospy.Time.now()

    def publish_image(self, image, image_publisher, timestamp):
        """This image publishes the `image` to the `image_publisher` with the given `timestamp`."""
        image_ros = self.bridge.cv2_to_imgmsg(image, "rgb8")
        image_ros.header.stamp = timestamp
        image_publisher.publish(image_ros)

    def image_callback(self, subject_image_list):
        """This method is called whenever new input arrives. The input is first converted in a format suitable
        for the gaze estimation network (see :meth:`input_from_image`), then the gaze is estimated (see
        :meth:`estimate_gaze`. The estimated gaze is overlaid on the input image (see :meth:`visualize_eye_result`),
        and this image is published along with the estimated gaze vector (see :meth:`publish_image` and
        :func:`publish_gaze`)"""
        timestamp = subject_image_list.header.stamp
        subjects_gaze_img = None

        subjects_dict = self.subjects_bridge.msg_to_images(subject_image_list)
        input_r_list = []
        input_l_list = []
        input_head_list = []
        valid_subject_list = []
        for subject_id, s in subjects_dict.items():
            try:
                (trans_head, rot_head) = self.tf_listener.lookupTransform(self.rgb_frame_id_ros, self.headpose_frame + str(subject_id), timestamp)
                euler_angles_head = gaze_tools.limit_yaw(rot_head)

                phi_head, theta_head = gaze_tools.get_phi_theta_from_euler(euler_angles_head)
                input_head_list.append([theta_head, phi_head])
                input_r_list.append(self.input_from_image(s.right))
                input_l_list.append(self.input_from_image(s.left))
                valid_subject_list.append(subject_id)
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException, tf.Exception):
                pass

        gaze_est = self.estimate_gaze_twoeyes(inference_input_left_list=input_l_list, inference_input_right_list=input_r_list, inference_headpose_list=input_head_list)

        for subject_id, gaze in zip(valid_subject_list, gaze_est.tolist()):
            s = subjects_dict[subject_id]
            r_gaze_img = self.visualize_eye_result(s.right, gaze)
            l_gaze_img = self.visualize_eye_result(s.left, gaze)
            s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)
            if subjects_gaze_img is None:
                subjects_gaze_img = s_gaze_img
            else:
                subjects_gaze_img = np.concatenate((subjects_gaze_img, s_gaze_img), axis=0)

            self.publish_gaze(gaze, timestamp, subject_id)

        if subjects_gaze_img is not None:
            gaze_img_msg = self.bridge.cv2_to_imgmsg(subjects_gaze_img.astype(np.uint8), "bgr8")
            self.subjects_gaze_img.publish(gaze_img_msg)

    def publish_gaze(self, est_gaze, msg_stamp, subject_id):
        """Publish the gaze vector as a PointStamped."""
        theta_gaze = est_gaze[0]
        phi_gaze = est_gaze[1]
        euler_angle_gaze = gaze_tools.get_euler_from_phi_theta(phi_gaze, theta_gaze)
        quaternion_gaze = tf.transformations.quaternion_from_euler(*euler_angle_gaze)
        self.tf_broadcaster.sendTransform((0, 0, 0.05),  # publish it 5cm above the head pose's origin (nose tip)
                                          quaternion_gaze, msg_stamp, self.tf_prefix + "/world_gaze" + str(subject_id), self.headpose_frame + str(subject_id))


if __name__ == '__main__':
    try:
        rospy.init_node('estimate_gaze')
        gaze_estimator = GazeEstimatorROS(rospy.get_param("~device_id_gazeestimation", default="/gpu:0"),
                                          [os.path.join(rospkg.RosPack().get_path('rt_gene'), model_file) for model_file in rospy.get_param("~model_files")])
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
