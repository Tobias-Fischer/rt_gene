#!/usr/bin/env python

"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
"""

from __future__ import print_function, division, absolute_import

import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from tqdm import tqdm
from image_geometry import PinholeCameraModel
import tf.transformations as tf_transformations
from geometry_msgs.msg import PointStamped, Point
from tf import TransformBroadcaster, TransformListener, ExtrapolationException
import time

# noinspection PyUnresolvedReferences
import rt_gene.gaze_tools as gaze_tools

# noinspection PyUnresolvedReferences
from rt_gene.kalman_stabilizer import Stabilizer

from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.subject_ros_bridge import SubjectListBridge

class SubjectDetected:
    def __init__(self, face_bb):
        self.face_bb = face_bb
        self.landmark_points = None
        self.left_eye_bb = None
        self.right_eye_bb = None

class LandmarkMethod(object):
    def __init__(self):
        self.subjects = dict()
        self.bridge = CvBridge()
        self.__subject_bridge = SubjectListBridge()

        self.margin = rospy.get_param("~margin", 42)
        self.margin_eyes_height = rospy.get_param("~margin_eyes_height", 36)
        self.margin_eyes_width = rospy.get_param("~margin_eyes_width", 60)
        self.interpupillary_distance = rospy.get_param("~interpupillary_distance", default=0.058)
        self.cropped_face_size = (rospy.get_param("~face_size_height", 224), rospy.get_param("~face_size_width", 224))

        self.rgb_frame_id = rospy.get_param("~rgb_frame_id", "/kinect2_link")
        self.rgb_frame_id_ros = rospy.get_param("~rgb_frame_id_ros", "/kinect2_nonrotated_link")

        self.model_points = None
        self.eye_image_size = (rospy.get_param("~eye_image_height", 36), rospy.get_param("~eye_image_width", 60))

        self.tf_broadcaster = TransformBroadcaster()
        self.tf_listener = TransformListener()
        self.tf_prefix = rospy.get_param("~tf_prefix", default="gaze")

        self.use_previous_headpose_estimate = True
        self.last_rvec = {}
        self.last_tvec = {}
        self.pose_stabilizers = {}  # Introduce scalar stabilizers for pose.

        try:
            tqdm.write("Wait for camera message")
            cam_info = rospy.wait_for_message("/camera_info", CameraInfo, timeout=None)
            self.img_proc = PinholeCameraModel()
            # noinspection PyTypeChecker
            self.img_proc.fromCameraInfo(cam_info)
            if np.array_equal(self.img_proc.intrinsicMatrix(), np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])):
                raise Exception('Camera matrix is zero-matrix. Did you calibrate'
                                'the camera and linked to the yaml file in the launch file?')
            tqdm.write("Camera message received")
        except rospy.ROSException:
            raise Exception("Could not get camera info")

        # multiple person images publication
        self.subject_pub = rospy.Publisher("/subjects/images", MSG_SubjectImagesList, queue_size=1)
        # multiple person faces publication for visualisation
        self.subject_faces_pub = rospy.Publisher("/subjects/faces", Image, queue_size=1)

    def process_image(self, color_msg):
        tqdm.write('Time now: ' + str(rospy.Time.now().to_sec())
                   + ' message color: ' + str(color_msg.header.stamp.to_sec())
                   + ' diff: ' + str(rospy.Time.now().to_sec() - color_msg.header.stamp.to_sec()))

        start_time = time.time()

        color_img = gaze_tools.convert_image(color_msg, "rgb8")
        timestamp = color_msg.header.stamp

        self.detect_landmarks(color_img, timestamp)  # update self.subjects

        tqdm.write('Elapsed after detecting transformed_landmarks: ' + str(time.time() - start_time))

        if not self.subjects:
            tqdm.write("No face found")
            return

        self.get_eye_image()    # update self.subjects

        final_head_pose_images = None
        for subject_id, subject in self.subjects.items():
            if subject.left_eye_color is None or subject.right_eye_color is None:
                continue
            if subject_id not in self.last_rvec:
                self.last_rvec[subject_id] = np.array([[0.01891013], [0.08560084], [-3.14392813]])
            if subject_id not in self.last_tvec:
                self.last_tvec[subject_id] = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])
            if subject_id not in self.pose_stabilizers:
                self.pose_stabilizers[subject_id] = [Stabilizer(
                    state_num=2,
                    measure_num=1,
                    cov_process=0.1,
                    cov_measure=0.1) for _ in range(6)]

            success, rotation_vector, translation_vector = self.get_head_pose(color_img, subject.marks, subject_id)

            # Publish all the data
            translation_headpose_tf = self.get_head_translation(timestamp, subject_id)

            if success:
                if translation_headpose_tf is not None:
                    euler_angles_head = self.publish_pose(timestamp, translation_headpose_tf, rotation_vector, subject_id)

                    if euler_angles_head is not None:
                        headpose_image = self.visualize_headpose_result(subject.face_color, gaze_tools.get_phi_theta_from_euler(euler_angles_head))
                        
                        if final_head_pose_images is None:
                            final_head_pose_images = headpose_image
                        else:
                            final_head_pose_images = np.concatenate((final_head_pose_images, headpose_image), axis=1)
            else:
                if not success:
                    tqdm.write("Could not get head pose properly")

        if final_head_pose_images is not None:
            self.publish_subject_list(timestamp, self.subjects)
            headpose_image_ros = self.bridge.cv2_to_imgmsg(final_head_pose_images, "bgr8")
            headpose_image_ros.header.stamp = timestamp
            self.subject_faces_pub.publish(headpose_image_ros)
                
        tqdm.write('Elapsed total: ' + str(time.time() - start_time) + '\n\n')

    # noinspection PyUnusedLocal
    def get_head_pose(self, color_img, landmarks, subject_id):
        """
        We are given a set of 2D points in the form of landmarks. The basic idea is that we assume a generic 3D head
        model, and try to fit the 2D points to the 3D model based on the Levenberg-Marquardt optimization. We can use
        OpenCV's implementation of SolvePnP for this.
        This method is inspired by http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        We are planning to replace this with our own head pose estimator.
        :param color_img: RGB Image
        :param landmarks: Landmark positions to be used to determine head pose
        :return: success - Whether the pose was successfully extracted
                 rotation_vector - rotation vector that along with translation vector brings points from the model
                 coordinate system to the camera coordinate system
                 translation_vector - see rotation_vector
        """

        image_points_headpose = self.get_image_points_headpose(landmarks)

        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()

        # tqdm.write("Camera Matrix :\n {0}".format(camera_matrix))

        try:
            if self.last_rvec[subject_id] is not None and self.last_tvec[subject_id] is not None and self.use_previous_headpose_estimate:
                (success, rotation_vector_unstable, translation_vector_unstable) = \
                    cv2.solvePnP(self.model_points, image_points_headpose, camera_matrix, dist_coeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True,
                                 rvec=self.last_rvec[subject_id], tvec=self.last_tvec[subject_id])
            else:
                (success, rotation_vector_unstable, translation_vector_unstable) = \
                    cv2.solvePnP(self.model_points, image_points_headpose, camera_matrix, dist_coeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        except Exception:
            print('Could not estimate head pose')
            return False, None, None

        if not success:
            print('Could not estimate head pose')
            return False, None, None

        # Apply Kalman filter
        stable_pose = []
        pose_np = np.array((rotation_vector_unstable, translation_vector_unstable)).flatten()
        for value, ps_stb in zip(pose_np, self.pose_stabilizers[subject_id]):
            ps_stb.update([value])
            stable_pose.append(ps_stb.state[0])

        stable_pose = np.reshape(stable_pose, (-1, 3))
        rotation_vector = stable_pose[0]
        translation_vector = stable_pose[1]

        self.last_rvec[subject_id] = rotation_vector
        self.last_tvec[subject_id] = translation_vector

        rotation_vector_swapped = [-rotation_vector[2], -rotation_vector[1] + np.pi, rotation_vector[0]]
        rot_head = tf_transformations.quaternion_from_euler(*rotation_vector_swapped)

        print('rot_head', rot_head, rotation_vector_unstable)

        return success, rot_head, translation_vector

    def detect_landmarks(self, color_img, timestamp):
        raise NotImplementedError('Detect_landmarks: Abstract class')

    def get_eye_image(self):
        raise NotImplementedError('Detect_landmarks: Abstract class')

    @staticmethod
    def get_image_points_headpose(landmarks):
        raise NotImplementedError('Detect_landmarks: Abstract class')

    @staticmethod
    def get_image_points_eyes_nose(landmarks):
        raise NotImplementedError('Detect_landmarks: Abstract class')

    def publish_subject_list(self, timestamp, subjects):
        assert(subjects is not None)
        
        subject_list_message = self.__subject_bridge.images_to_msg(subjects, timestamp)
            
        self.subject_pub.publish(subject_list_message)

    @staticmethod
    def visualize_headpose_result(face_image, est_headpose):
        """Here, we take the original eye eye_image and overlay the estimated gaze."""
        output_image = np.copy(face_image)

        center_x = face_image.shape[1] / 2
        center_y = face_image.shape[0] / 2

        endpoint_x, endpoint_y = gaze_tools.get_endpoint(est_headpose[1], est_headpose[0], center_x, center_y, 100)

        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (0, 0, 255), 3)
        return output_image

    def get_head_translation(self, timestamp, subject_id):
        trans_reshaped = self.last_tvec[subject_id].reshape(3, 1)
        nose_center_3d_rot = [-float(trans_reshaped[0] / 1000.0),
                              -float(trans_reshaped[1] / 1000.0),
                              -float(trans_reshaped[2] / 1000.0)]

        nose_center_3d_rot_frame = self.rgb_frame_id

        try:
            nose_center_3d_rot_pt = PointStamped()
            nose_center_3d_rot_pt.header.frame_id = nose_center_3d_rot_frame
            nose_center_3d_rot_pt.header.stamp = timestamp
            nose_center_3d_rot_pt.point = Point(x=nose_center_3d_rot[0],
                                                y=nose_center_3d_rot[1],
                                                z=nose_center_3d_rot[2])
            nose_center_3d = self.tf_listener.transformPoint(self.rgb_frame_id_ros, nose_center_3d_rot_pt)
            nose_center_3d.header.stamp = timestamp

            nose_center_3d_tf = gaze_tools.position_ros_to_tf(nose_center_3d.point)

            print('Translation based on landmarks', nose_center_3d_tf)
            return nose_center_3d_tf
        except ExtrapolationException as e:
            print(e)
            return None

    def publish_pose(self, timestamp, nose_center_3d_tf, rot_head, subject_id):
        self.tf_broadcaster.sendTransform(nose_center_3d_tf,
                                          rot_head,
                                          timestamp,
                                          self.tf_prefix + "/head_pose_estimated" + str(subject_id),
                                          self.rgb_frame_id_ros)

        return gaze_tools.get_head_pose(nose_center_3d_tf, rot_head)

    def callback(self, color_msg):
        """Simply call process_image."""
        try:
            self.process_image(color_msg)

        except CvBridgeError as e:
            print(e)
        except rospy.ROSException as e:
            if str(e) == "publish() to a closed topic":
                pass
            else:
                raise e

