#!/usr/bin/env python

"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import cv2
from cv_bridge import CvBridge
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from sensor_msgs.msg import Image, CameraInfo
from tqdm import tqdm
from image_geometry import PinholeCameraModel
import tf.transformations as tf_transformations
from geometry_msgs.msg import PointStamped, Point
from tf import TransformBroadcaster, TransformListener, ExtrapolationException, transformations
from dynamic_reconfigure.server import Server
import rospy

import numpy as np

import rt_gene.gaze_tools as gaze_tools

from rt_gene.kalman_stabilizer import Stabilizer

from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.cfg import ModelSizeConfig
from rt_gene.subject_ros_bridge import SubjectListBridge
from rt_gene.tracker_face_encoding import FaceEncodingTracker
from rt_gene.tracker_sequential import SequentialTracker


class LandmarkMethod(LandmarkMethodBase):
    def __init__(self, img_proc=None):
        super(LandmarkMethod, self).__init__(device_id_facedetection=rospy.get_param("~device_id_facedetection", default="cuda:0"))
        self.subject_tracker = FaceEncodingTracker() if rospy.get_param("~use_face_encoding_tracker", default=True) else SequentialTracker()
        self.bridge = CvBridge()
        self.__subject_bridge = SubjectListBridge()

        self.ros_tf_frame = rospy.get_param("~ros_tf_frame", "/kinect2_nonrotated_link")

        self.tf_broadcaster = TransformBroadcaster()
        self.tf_listener = TransformListener()
        self.tf_prefix = rospy.get_param("~tf_prefix", default="gaze")

        self.use_previous_headpose_estimate = True
        self.pose_stabilizers = {}  # Introduce scalar stabilizers for pose.

        try:
            if img_proc is None:
                tqdm.write("Wait for camera message")
                cam_info = rospy.wait_for_message("/camera_info", CameraInfo, timeout=None)
                self.img_proc = PinholeCameraModel()
                # noinspection PyTypeChecker
                self.img_proc.fromCameraInfo(cam_info)
            else:
                self.img_proc = img_proc

            if np.array_equal(self.img_proc.intrinsicMatrix().A, np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])):
                raise Exception('Camera matrix is zero-matrix. Did you calibrate'
                                'the camera and linked to the yaml file in the launch file?')
            tqdm.write("Camera message received")
        except rospy.ROSException:
            raise Exception("Could not get camera info")

        # multiple person images publication
        self.subject_pub = rospy.Publisher("/subjects/images", MSG_SubjectImagesList, queue_size=1)
        # multiple person faces publication for visualisation
        self.subject_faces_pub = rospy.Publisher("/subjects/faces", Image, queue_size=1)

        Server(ModelSizeConfig, self._dyn_reconfig_callback)

        # have the subscriber the last thing that's run to avoid conflicts
        self.color_sub = rospy.Subscriber("/image", Image, self.process_image, buff_size=2 ** 24, queue_size=1)

    def _dyn_reconfig_callback(self, config, _):
        self.model_points /= (self.model_size_rescale * self.interpupillary_distance)
        self.model_size_rescale = config["model_size"]
        self.interpupillary_distance = config["interpupillary_distance"]
        self.model_points *= (self.model_size_rescale * self.interpupillary_distance)
        self.head_pitch = config["head_pitch"]
        return config

    def process_image(self, color_msg):
        tqdm.write('Time now: {} message color: {} diff: {:.2f}s'.format((rospy.Time.now().to_sec()), color_msg.header.stamp.to_sec(),
                                                                         rospy.Time.now().to_sec() - color_msg.header.stamp.to_sec()))

        color_img = gaze_tools.convert_image(color_msg, "bgr8")
        timestamp = color_msg.header.stamp

        self.update_subject_tracker(color_img)

        if not self.subject_tracker.get_tracked_elements():
            tqdm.write("No face found")
            return

        self.subject_tracker.update_eye_images(self.eye_image_size)

        final_head_pose_images = None
        for subject_id, subject in self.subject_tracker.get_tracked_elements().items():
            if subject.left_eye_color is None or subject.right_eye_color is None:
                continue
            if subject_id not in self.pose_stabilizers:
                self.pose_stabilizers[subject_id] = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(6)]

            success, head_rpy, translation_vector = self.get_head_pose(subject.marks, subject_id)

            if success:
                # Publish all the data
                self.publish_pose(timestamp, translation_vector, head_rpy, subject_id)

                roll_pitch_yaw = gaze_tools.limit_yaw(head_rpy)
                face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

                head_pose_image = LandmarkMethod.visualize_headpose_result(face_image_resized, gaze_tools.get_phi_theta_from_euler(roll_pitch_yaw))

                if final_head_pose_images is None:
                    final_head_pose_images = head_pose_image
                else:
                    final_head_pose_images = np.concatenate((final_head_pose_images, head_pose_image), axis=1)
            else:
                tqdm.write("Could not get head pose properly")

        if final_head_pose_images is not None:
            self.publish_subject_list(timestamp, self.subject_tracker.get_tracked_elements())
            headpose_image_ros = self.bridge.cv2_to_imgmsg(final_head_pose_images, "bgr8")
            headpose_image_ros.header.stamp = timestamp
            self.subject_faces_pub.publish(headpose_image_ros)

    def get_head_pose(self, landmarks, subject_id):
        """
        We are given a set of 2D points in the form of landmarks. The basic idea is that we assume a generic 3D head
        model, and try to fit the 2D points to the 3D model based on the Levenberg-Marquardt optimization. We can use
        OpenCV's implementation of SolvePnP for this.
        This method is inspired by http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        We are planning to replace this with our own head pose estimator.
        :param landmarks: Landmark positions to be used to determine head pose
        :param subject_id: ID of the subject that corresponds to the given landmarks
        :return: success - Whether the pose was successfully extracted
                 rotation_vector - rotation vector that along with translation vector brings points from the model
                 coordinate system to the camera coordinate system
                 translation_vector - see rotation_vector
        """

        image_points_headpose = landmarks

        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()

        try:
            success, rotation_vector_unstable, translation_vector_unstable = cv2.solvePnP(self.model_points,
                                                                                          image_points_headpose.reshape(len(self.model_points), 1, 2),
                                                                                          cameraMatrix=camera_matrix,
                                                                                          distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_DLS)
        except cv2.error as e:
            print('Could not estimate head pose', e)
            return False, None, None

        if not success:
            print('Could not estimate head pose')
            return False, None, None

        rotation_vector, translation_vector = self.apply_kalman_filter_head_pose(subject_id, rotation_vector_unstable, translation_vector_unstable)

        translation_vector = np.array([translation_vector[2], -translation_vector[0], -translation_vector[1]]) / 1000.0
        rotation_vector_swapped = [-rotation_vector[2], -rotation_vector[0] + self.head_pitch, rotation_vector[1] + np.pi]

        return success, rotation_vector_swapped, translation_vector

    def apply_kalman_filter_head_pose(self, subject_id, rotation_vector_unstable, translation_vector_unstable):
        stable_pose = []
        pose_np = np.array((rotation_vector_unstable, translation_vector_unstable)).flatten()
        for value, ps_stb in zip(pose_np, self.pose_stabilizers[subject_id]):
            ps_stb.update([value])
            stable_pose.append(ps_stb.state[0])
        stable_pose = np.reshape(stable_pose, (-1, 3))
        rotation_vector = stable_pose[0]
        translation_vector = stable_pose[1]
        return rotation_vector, translation_vector

    def publish_subject_list(self, timestamp, subjects):
        assert (subjects is not None)

        subject_list_message = self.__subject_bridge.images_to_msg(subjects, timestamp)

        self.subject_pub.publish(subject_list_message)

    def publish_pose(self, timestamp, nose_center_3d_tf, head_rpy, subject_id):
        self.tf_broadcaster.sendTransform(nose_center_3d_tf, transformations.quaternion_from_euler(*head_rpy), timestamp,
                                          self.tf_prefix + "/head_pose_estimated" + str(subject_id),
                                          self.ros_tf_frame)

    def update_subject_tracker(self, color_img):
        faceboxes = self.get_face_bb(color_img)
        if len(faceboxes) == 0:
            self.subject_tracker.clear_elements()
            return

        tracked_subjects = self.get_subjects_from_faceboxes(color_img, faceboxes)

        # track the new faceboxes according to the previous ones
        self.subject_tracker.track(tracked_subjects)
