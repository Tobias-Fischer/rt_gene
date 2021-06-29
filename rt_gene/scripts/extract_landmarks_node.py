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
from geometry_msgs.msg import TransformStamped
from tqdm import tqdm
from image_geometry import PinholeCameraModel
from tf2_ros import TransformBroadcaster, TransformListener, Buffer
from tf import transformations
from dynamic_reconfigure.server import Server
import rospy

import numpy as np

import rt_gene.gaze_tools as gaze_tools
import rt_gene.ros_tools as ros_tools

from rt_gene.kalman_stabilizer import Stabilizer

from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.msg import MSG_Headpose, MSG_HeadposeList
from rt_gene.msg import MSG_Landmarks, MSG_LandmarksList

from rt_gene.cfg import ModelSizeConfig
from rt_gene.subject_ros_bridge import SubjectListBridge
from rt_gene.tracker_face_encoding import FaceEncodingTracker
from rt_gene.tracker_sequential import SequentialTracker


class LandmarkMethodROS(LandmarkMethodBase):
    def __init__(self, img_proc=None):
        super(LandmarkMethodROS, self).__init__(device_id_facedetection=rospy.get_param("~device_id_facedetection", default="cuda:0"))
        self.subject_tracker = FaceEncodingTracker() if rospy.get_param("~use_face_encoding_tracker", default=True) else SequentialTracker()
        self.bridge = CvBridge()
        self.__subject_bridge = SubjectListBridge()

        self.tf2_broadcaster = TransformBroadcaster()
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer)
        self.tf_prefix = rospy.get_param("~tf_prefix", default="gaze")
        self.visualise_headpose = rospy.get_param("~visualise_headpose", default=True)
        self._pnp_iterate_after = rospy.get_param("~pnp_iterate_after", default=False)

        self.pose_stabilizers = {}  # Introduce scalar stabilizers for pose.

        try:
            if img_proc is None:
                tqdm.write("Wait for camera message")
                cam_info = rospy.wait_for_message("/camera_info", CameraInfo, timeout=None)
                self.img_proc = PinholeCameraModel()
                # noinspection PyTypeChecker
                self.img_proc.fromCameraInfo(cam_info)
                self.camera_frame = cam_info.header.frame_id
                if self.camera_frame.startswith("/"):
                    self.camera_frame = self.camera_frame[1:]
            else:
                self.img_proc = img_proc

            if np.array_equal(self.img_proc.intrinsicMatrix().A, np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])):
                raise Exception('Camera matrix is zero-matrix. Did you calibrate'
                                'the camera and linked to the yaml file in the launch file?')
            tqdm.write("Camera message received")
        except rospy.ROSException:
            raise Exception("Could not get camera info")

        # multiple person images publication
        self.subject_pub = rospy.Publisher("/subjects/images", MSG_SubjectImagesList, queue_size=3)
        self.headpose_publisher = rospy.Publisher("/subjects/headpose", MSG_HeadposeList, queue_size=3)
        self.landmark_publisher = rospy.Publisher("/subjects/landmarks", MSG_LandmarksList, queue_size=3)
        # multiple person faces publication for visualisation
        self.subject_faces_pub = rospy.Publisher("/subjects/faces", Image, queue_size=3)

        Server(ModelSizeConfig, self._dyn_reconfig_callback)

        # have the subscriber the last thing that's run to avoid conflicts
        self.color_sub = rospy.Subscriber("/image", Image, self.process_image, buff_size=2 ** 24, queue_size=3)

    def _dyn_reconfig_callback(self, config, _):
        self.model_points /= (self.model_size_rescale * self.interpupillary_distance)
        self.model_size_rescale = config["model_size"]
        self.interpupillary_distance = config["interpupillary_distance"]
        self.model_points *= (self.model_size_rescale * self.interpupillary_distance)
        self.head_pitch = config["head_pitch"]
        return config

    def process_image(self, color_msg):
        color_img = ros_tools.convert_image(color_msg, "bgr8")
        timestamp = color_msg.header.stamp

        self.update_subject_tracker(color_img)

        if not self.subject_tracker.get_tracked_elements():
            tqdm.write("\033[2K\033[1;31mNo face found\033[0m", end="\r")
            return

        self.subject_tracker.update_eye_images(self.eye_image_size)

        final_head_pose_images = []
        for subject_id, subject in self.subject_tracker.get_tracked_elements().items():
            if subject.left_eye_color is None or subject.right_eye_color is None:
                continue
            if subject_id not in self.pose_stabilizers:
                self.pose_stabilizers[subject_id] = [Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1) for _ in range(6)]

            success, head_rpy, translation_vector = self.get_head_pose(subject.landmarks, subject_id)

            if success:
                # Publish all the data
                subject.head_rotation = head_rpy
                subject.head_translation = translation_vector
                self.publish_pose(timestamp, translation_vector, head_rpy, subject_id)

                if self.visualise_headpose:
                    roll_pitch_yaw = list(transformations.euler_from_matrix(np.dot(ros_tools.camera_to_ros, transformations.euler_matrix(*head_rpy))))
                    roll_pitch_yaw = gaze_tools.limit_yaw(roll_pitch_yaw)

                    face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

                    final_head_pose_images.append(
                        LandmarkMethodROS.visualize_headpose_result(face_image_resized, gaze_tools.get_phi_theta_from_euler(roll_pitch_yaw)))

        if len(self.subject_tracker.get_tracked_elements().items()) > 0:
            self.publish_subject_list(timestamp, self.subject_tracker.get_tracked_elements())

        if len(final_head_pose_images) > 0:
            headpose_image_ros = self.bridge.cv2_to_imgmsg(np.hstack(final_head_pose_images), "bgr8")
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

        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()

        try:
            success, rodrigues_rotation, translation_vector, _ = cv2.solvePnPRansac(self.model_points,
                                                                                    landmarks.reshape(len(self.model_points), 1, 2),
                                                                                    cameraMatrix=camera_matrix,
                                                                                    distCoeffs=dist_coeffs, flags=cv2.SOLVEPNP_DLS)

            if self._pnp_iterate_after:
                success, rodrigues_rotation, translation_vector = cv2.solvePnP(self.model_points,
                                                                               landmarks.reshape(len(self.model_points), 1, 2),
                                                                               rvec=rodrigues_rotation,
                                                                               tvec=translation_vector,
                                                                               useExtrinsicGuess=True,
                                                                               cameraMatrix=camera_matrix,
                                                                               distCoeffs=dist_coeffs,
                                                                               flags=cv2.SOLVEPNP_ITERATIVE)


        except cv2.error as e:
            tqdm.write('\033[2K\033[1;31mCould not estimate head pose: {}\033[0m'.format(e), end="\r")
            return False, None, None

        if not success:
            tqdm.write('\033[2K\033[1;31mUnsuccessful in solvingPnPRanscan\033[0m', end="\r")
            return False, None, None

        # this is generic point stabiliser, the underlying representation doesn't matter
        rotation_vector, translation_vector = self.apply_kalman_filter_head_pose(subject_id, rodrigues_rotation, translation_vector / 1000.0)

        rotation_vector[0] += self.head_pitch

        _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        _rpy_rotation = np.array(transformations.euler_from_matrix(_m))

        return success, _rpy_rotation, translation_vector

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
        subject_list_message.header.frame_id = self.camera_frame
        self.subject_pub.publish(subject_list_message)

        landmark_msg_list = MSG_LandmarksList()
        landmark_msg_list.header.stamp = timestamp
        landmark_msg_list.header.frame_id = self.camera_frame

        headpose_msg_list = MSG_HeadposeList()
        headpose_msg_list.header.stamp = timestamp
        headpose_msg_list.header.frame_id = self.camera_frame

        for subject_id, s in subjects.items():
            try:
                landmark_msg = MSG_Landmarks()
                landmark_msg.subject_id = str(subject_id)
                landmark_msg.landmarks = s.landmarks.flatten()
                landmark_msg_list.subjects.append(landmark_msg)
            except AttributeError:
                # we haven't assigned landmarks to the subject "s" yet...ignore this subject for the time being
                pass

            try:
                headpose_msg = MSG_Headpose()
                headpose_msg.subject_id = subject_id
                headpose_msg.roll = s.head_rotation[0]
                headpose_msg.pitch = s.head_rotation[1]
                headpose_msg.yaw = s.head_rotation[2]
                headpose_msg.x = s.head_translation[0]
                headpose_msg.y = s.head_translation[1]
                headpose_msg.z = s.head_translation[2]
                headpose_msg_list.subjects.append(headpose_msg)
            except AttributeError:
                # we haven't assigned landmarks to the subject "s" yet...ignore this subject for the time being
                pass

        self.landmark_publisher.publish(landmark_msg_list)
        self.headpose_publisher.publish(headpose_msg_list)

    def publish_pose(self, timestamp, nose_center_3d_tf, head_rpy, subject_id):
        t = TransformStamped()
        t.header.frame_id = self.camera_frame
        t.header.stamp = timestamp
        t.child_frame_id = self.tf_prefix + "/head_pose_estimated" + str(subject_id)
        t.transform.translation.x = nose_center_3d_tf[0]
        t.transform.translation.y = nose_center_3d_tf[1]
        t.transform.translation.z = nose_center_3d_tf[2]

        rotation = transformations.quaternion_from_euler(*head_rpy)
        t.transform.rotation.x = rotation[0]
        t.transform.rotation.y = rotation[1]
        t.transform.rotation.z = rotation[2]
        t.transform.rotation.w = rotation[3]

        try:
            self.tf2_broadcaster.sendTransform([t])
        except rospy.ROSException as exc:
            if str(exc) == "publish() to a closed topic":
                pass
            else:
                raise exc

    def update_subject_tracker(self, color_img):
        faceboxes = self.get_face_bb(color_img)
        if len(faceboxes) == 0:
            self.subject_tracker.clear_elements()
            return

        tracked_subjects = self.get_subjects_from_faceboxes(color_img, faceboxes)

        # track the new faceboxes according to the previous ones
        self.subject_tracker.track(tracked_subjects)


if __name__ == '__main__':
    try:
        rospy.init_node('extract_landmarks')

        landmark_extractor = LandmarkMethodROS()

        rospy.spin()
    except rospy.exceptions.ROSInterruptException:
        print("See ya")
    except KeyboardInterrupt:
        print("Shutting down")
