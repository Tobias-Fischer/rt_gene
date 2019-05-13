#!/usr/bin/env python

"""
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Kevin Cortacero <cortacero.k31130@gmail.com>
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

Uses face-alignment package (https://github.com/1adrianb/face-alignment)
"""

from __future__ import print_function, division, absolute_import

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo
from tqdm import tqdm
from image_geometry import PinholeCameraModel
import tf.transformations as tf_transformations
from geometry_msgs.msg import PointStamped, Point
from tf import TransformBroadcaster, TransformListener, ExtrapolationException
import scipy
from dynamic_reconfigure.server import Server
import time
import rospkg
import rospy

import numpy as np

import rt_gene.gaze_tools as gaze_tools

from rt_gene.kalman_stabilizer import Stabilizer

from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.cfg import ModelSizeConfig
from rt_gene.subject_ros_bridge import SubjectListBridge

import face_alignment
from face_alignment.detection.sfd import FaceDetector

import torch


class SubjectDetected(object):
    def __init__(self, face_bb):
        self.face_bb = face_bb
        self.landmark_points = None
        self.left_eye_bb = None
        self.right_eye_bb = None


class LandmarkMethod(object):
    def __init__(self, img_proc=None):
        self.subjects = dict()
        self.bridge = CvBridge()
        self.__subject_bridge = SubjectListBridge()
        self.model_size_rescale = 30.0
        self.head_pitch = 0.0
        self.margin = rospy.get_param("~margin", 42)
        self.margin_eyes_height = rospy.get_param("~margin_eyes_height", 36)
        self.margin_eyes_width = rospy.get_param("~margin_eyes_width", 60)
        self.interpupillary_distance = 0.058
        self.cropped_face_size = (rospy.get_param("~face_size_height", 224), rospy.get_param("~face_size_width", 224))

        self.device_id_facedetection = rospy.get_param("~device_id_facedetection", default="cuda:0")
        self.device_id_facealignment = rospy.get_param("~device_id_facealignment", default="cuda:0")
        rospy.loginfo("Using device {} for face detection.".format(self.device_id_facedetection))
        rospy.loginfo("Using device {} for face alignment.".format(self.device_id_facealignment))

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
            if img_proc is None:
                tqdm.write("Wait for camera message")
                cam_info = rospy.wait_for_message("/camera_info", CameraInfo, timeout=None)
                self.img_proc = PinholeCameraModel()
                # noinspection PyTypeChecker
                self.img_proc.fromCameraInfo(cam_info)
            else:
                self.img_proc = img_proc

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

        self.model_points = self._get_full_model_points()

        self.sess_bb = None
        self.face_net = FaceDetector(device=self.device_id_facedetection)

        self.facial_landmark_nn = face_alignment.FaceAlignment(landmarks_type=face_alignment.LandmarksType._2D,
                                                               device=self.device_id_facealignment, flip_input=False)

        Server(ModelSizeConfig, self._dyn_reconfig_callback)

        # have the subscriber the last thing that's run to avoid conflicts
        self.color_sub = rospy.Subscriber("/image", Image, self.callback, buff_size=2 ** 24, queue_size=1)

    def _dyn_reconfig_callback(self, config, level):
        self.model_points /= (self.model_size_rescale * self.interpupillary_distance)
        self.model_size_rescale = config["model_size"]
        self.interpupillary_distance = config["interpupillary_distance"]
        self.model_points *= (self.model_size_rescale * self.interpupillary_distance)
        self.head_pitch = config["head_pitch"]
        return config

    def _get_full_model_points(self):
        """Get all 68 3D model points from file"""
        raw_value = []
        filename = rospkg.RosPack().get_path('rt_gene') + '/model_nets/face_model_68.txt'
        with open(filename) as f:
            for line in f:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T
        # model_points *= 4
        model_points[:, -1] *= -1

        # index the expansion of the model based.
        model_points = model_points * (self.interpupillary_distance * self.model_size_rescale)

        return model_points

    def get_face_bb(self, image):
        faceboxes = []

        start_time = time.time()
        fraction = 4.0
        image = scipy.misc.imresize(image, 1.0 / fraction)
        detections = self.face_net.detect_from_image(image)
        tqdm.write("Face Detector Frequency: {:.2f}Hz".format(1 / (time.time() - start_time)))

        for result in detections:
            # scale back up to image size
            x_left_top, y_left_top, x_right_bottom, y_right_bottom, confidence = result

            if x_left_top > 0 and y_left_top > 0 and x_right_bottom < image.shape[1] and y_right_bottom < image.shape[
                0] and confidence > 0.8:
                box = [x_left_top, y_left_top, x_right_bottom, y_right_bottom]
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = self.move_box(box, [0, offset_y])

                # Make box square.
                facebox = self.get_square_box(box_moved)
                faceboxes.append(facebox)

        return faceboxes

    @staticmethod
    def move_box(box, offset):
        """Move the box to direction specified by vector offset"""
        left_x = box[0] + offset[0]
        top_y = box[1] + offset[1]
        right_x = box[2] + offset[0]
        bottom_y = box[3] + offset[1]

        return [left_x, top_y, right_x, bottom_y]

    @staticmethod
    def box_in_image(box, image):
        """Check if the box is in image"""
        rows = image.shape[0]
        cols = image.shape[1]

        return box[0] >= 0 and box[1] >= 0 and box[2] <= cols and box[3] <= rows

    @staticmethod
    def get_square_box(box):
        """Get a square box out of the given box, by expanding it."""
        left_x = box[0]
        top_y = box[1]
        right_x = box[2]
        bottom_y = box[3]

        box_width = right_x - left_x
        box_height = bottom_y - top_y

        # Check if box is already a square. If not, make it a square.
        diff = box_height - box_width
        delta = int(abs(diff) / 2)

        if diff == 0:  # Already a square.
            return box
        elif diff > 0:  # Height > width, a slim box.
            left_x -= delta
            right_x += delta
            if diff % 2 == 1:
                right_x += 1
        else:  # Width > height, a short box.
            top_y -= delta
            bottom_y += delta
            if diff % 2 == 1:
                bottom_y += 1

        return [left_x, top_y, right_x, bottom_y]

    def process_image(self, color_msg):
        tqdm.write('Time now: ' + str(rospy.Time.now().to_sec())
                   + ' message color: ' + str(color_msg.header.stamp.to_sec())
                   + ' diff: ' + str(rospy.Time.now().to_sec() - color_msg.header.stamp.to_sec()))

        start_time = time.time()

        color_img = gaze_tools.convert_image(color_msg, "bgr8")
        timestamp = color_msg.header.stamp

        self.detect_landmarks(color_img, timestamp)  # update self.subjects

        tqdm.write('Elapsed after detecting transformed_landmarks: ' + str(time.time() - start_time))

        if not self.subjects:
            tqdm.write("No face found")
            return

        self.get_eye_image()  # update self.subjects

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

            success, rotation_vector, translation_vector = self.get_head_pose(subject.marks, subject_id)

            # Publish all the data
            translation_headpose_tf = self.get_head_translation(timestamp, subject_id)

            if success:
                if translation_headpose_tf is not None:
                    euler_angles_head = self.publish_pose(timestamp, translation_headpose_tf, rotation_vector,
                                                          subject_id)

                    if euler_angles_head is not None:
                        head_pose_image = self.visualize_headpose_result(subject.face_color,
                                                                         gaze_tools.get_phi_theta_from_euler(
                                                                             euler_angles_head))
                        head_pose_image_resized = cv2.resize(head_pose_image, dsize=(224, 224),
                                                             interpolation=cv2.INTER_CUBIC)

                        if final_head_pose_images is None:
                            final_head_pose_images = head_pose_image_resized
                        else:
                            final_head_pose_images = np.concatenate((final_head_pose_images, head_pose_image_resized),
                                                                    axis=1)
            else:
                if not success:
                    tqdm.write("Could not get head pose properly")

        if final_head_pose_images is not None:
            self.publish_subject_list(timestamp, self.subjects)
            headpose_image_ros = self.bridge.cv2_to_imgmsg(final_head_pose_images, "bgr8")
            headpose_image_ros.header.stamp = timestamp
            self.subject_faces_pub.publish(headpose_image_ros)

        tqdm.write('Elapsed total: ' + str(time.time() - start_time) + '\n\n')

        return self.subjects[0]

    def get_head_pose(self, landmarks, subject_id):
        """
        We are given a set of 2D points in the form of landmarks. The basic idea is that we assume a generic 3D head
        model, and try to fit the 2D points to the 3D model based on the Levenberg-Marquardt optimization. We can use
        OpenCV's implementation of SolvePnP for this.
        This method is inspired by http://www.learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        We are planning to replace this with our own head pose estimator.
        :param landmarks: Landmark positions to be used to determine head pose
        :return: success - Whether the pose was successfully extracted
                 rotation_vector - rotation vector that along with translation vector brings points from the model
                 coordinate system to the camera coordinate system
                 translation_vector - see rotation_vector
        """

        image_points_headpose = landmarks

        camera_matrix = self.img_proc.intrinsicMatrix()
        dist_coeffs = self.img_proc.distortionCoeffs()

        try:
            if self.last_rvec[subject_id] is not None and self.last_tvec[
                subject_id] is not None and self.use_previous_headpose_estimate:
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

        rotation_vector_swapped = [-rotation_vector[2], -rotation_vector[1] + np.pi + self.head_pitch,
                                   rotation_vector[0]]
        rot_head = tf_transformations.quaternion_from_euler(*rotation_vector_swapped)

        return success, rot_head, translation_vector

    def publish_subject_list(self, timestamp, subjects):
        assert (subjects is not None)

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
            print("** Extrapolation Exception **", e)
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

    def __update_subjects(self, new_faceboxes):
        """
        Assign the new faces to the existing subjects (id tracking)
        :param new_faceboxes: new faceboxes detected
        :return: update self.subjects
        """
        assert (self.subjects is not None)
        assert (new_faceboxes is not None)

        if len(new_faceboxes) == 0:
            self.subjects = dict()
            return

        if len(self.subjects) == 0:
            for j, b_new in enumerate(new_faceboxes):
                self.subjects[j] = SubjectDetected(b_new)
            return

        distance_matrix = np.ones((len(self.subjects), len(new_faceboxes)))
        for i, subject in enumerate(self.subjects.values()):
            for j, b_new in enumerate(new_faceboxes):
                distance_matrix[i][j] = np.sqrt(np.mean(((np.array(subject.face_bb) - np.array(b_new)) ** 2)))
        ids_to_assign = range(len(new_faceboxes))
        self.subjects = dict()
        for id in ids_to_assign:
            subject = np.argmin(distance_matrix[:, id])
            while subject in self.subjects:
                subject += 1
            self.subjects[subject] = SubjectDetected(new_faceboxes[id])

    def __detect_landmarks_one_box(self, facebox, color_img):
        try:
            _bb = map(int, facebox)
            face_img = color_img[_bb[1]: _bb[3], _bb[0]: _bb[2]]
            marks_orig = np.array(self.__detect_facial_landmarks(color_img, facebox)[0])

            eye_indices = np.array([36, 39, 42, 45])

            transformed_landmarks = marks_orig[eye_indices]
            transformed_landmarks[:, 0] -= facebox[0]
            transformed_landmarks[:, 1] -= facebox[1]

            return face_img, transformed_landmarks, marks_orig
        except Exception as e:
            print("*** Exception in detecting landmarks from facebox ***", e)
            return None, None, None

    def __detect_facial_landmarks(self, color_img, facebox):
        marks = self.facial_landmark_nn.get_landmarks(color_img, detected_faces=[facebox])
        return marks

    def detect_landmarks(self, color_img, timestamp):
        faceboxes = self.get_face_bb(color_img)

        self.__update_subjects(faceboxes)

        for subject in self.subjects.values():
            face, landmarks, marks = self.__detect_landmarks_one_box(subject.face_bb, color_img)
            subject.face_color = face
            subject.transformed_landmarks = landmarks
            subject.marks = marks

    def __get_eye_image_one(self, transformed_landmarks, face_aligned_color):
        margin_ratio = 1.0
        desired_ratio = float(self.eye_image_size[0]) / float(self.eye_image_size[1]) / 2.0

        try:
            # Get the width of the eye, and compute how big the margin should be according to the width
            lefteye_width = transformed_landmarks[3][0] - transformed_landmarks[2][0]
            righteye_width = transformed_landmarks[1][0] - transformed_landmarks[0][0]
            lefteye_margin, righteye_margin = lefteye_width * margin_ratio, righteye_width * margin_ratio

            # lefteye_center_x = transformed_landmarks[2][0] + lefteye_width / 2
            # righteye_center_x = transformed_landmarks[0][0] + righteye_width / 2
            lefteye_center_y = (transformed_landmarks[2][1] + transformed_landmarks[3][1]) / 2.0
            righteye_center_y = (transformed_landmarks[1][1] + transformed_landmarks[0][1]) / 2.0

            # Now compute the bounding boxes
            # The left / right x-coordinates are computed as the landmark position plus/minus the margin
            # The bottom / top y-coordinates are computed according to the desired ratio, as the width of the image is known
            left_bb = np.zeros(4, dtype=np.int)
            left_bb[0] = transformed_landmarks[2][0] - lefteye_margin / 2.0
            left_bb[1] = lefteye_center_y - (lefteye_width + lefteye_margin) * desired_ratio
            left_bb[2] = transformed_landmarks[3][0] + lefteye_margin / 2.0
            left_bb[3] = lefteye_center_y + (lefteye_width + lefteye_margin) * desired_ratio

            left_bb = map(int, left_bb)

            right_bb = np.zeros(4, dtype=np.float)
            right_bb[0] = transformed_landmarks[0][0] - righteye_margin / 2.0
            right_bb[1] = righteye_center_y - (righteye_width + righteye_margin) * desired_ratio
            right_bb[2] = transformed_landmarks[1][0] + righteye_margin / 2.0
            right_bb[3] = righteye_center_y + (righteye_width + righteye_margin) * desired_ratio

            right_bb = map(int, right_bb)

            # Extract the eye images from the aligned image
            left_eye_color = face_aligned_color[left_bb[1]:left_bb[3], left_bb[0]:left_bb[2], :]
            right_eye_color = face_aligned_color[right_bb[1]:right_bb[3], right_bb[0]:right_bb[2], :]

            # for p in transformed_landmarks:  # For debug visualization only
            #     cv2.circle(face_aligned_color, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            # So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
            left_eye_color_resized = scipy.misc.imresize(left_eye_color, self.eye_image_size, interp='cubic')
            right_eye_color_resized = scipy.misc.imresize(right_eye_color, self.eye_image_size, interp='cubic')

            return left_eye_color_resized, right_eye_color_resized, left_bb, right_bb
        except (ValueError, TypeError):
            return None, None, None, None

    # noinspection PyUnusedLocal
    def get_eye_image(self):
        """Extract the left and right eye images given the (dlib) transformed_landmarks and the source image.
        First, align the face. Then, extract the width of the eyes given the landmark positions.
        The height of the images is computed according to the desired ratio of the eye images."""

        start_time = time.time()
        for subject in self.subjects.values():
            le_c, re_c, le_bb, re_bb = self.__get_eye_image_one(subject.transformed_landmarks, subject.face_color)
            subject.left_eye_color = le_c
            subject.right_eye_color = re_c
            subject.left_eye_bb = le_bb
            subject.right_eye_bb = re_bb

        tqdm.write('New get_eye_image time: ' + str(time.time() - start_time))

    @staticmethod
    def get_image_points_eyes_nose(landmarks):
        landmarks_x, landmarks_y = landmarks.T[0], landmarks.T[1]

        left_eye_center_x = landmarks_x[42] + (landmarks_x[45] - landmarks_x[42]) / 2.0
        left_eye_center_y = (landmarks_y[42] + landmarks_y[45]) / 2.0
        right_eye_center_x = landmarks_x[36] + (landmarks_x[40] - landmarks_x[36]) / 2.0
        right_eye_center_y = (landmarks_y[36] + landmarks_y[40]) / 2.0
        nose_center_x, nose_center_y = (landmarks_x[33] + landmarks_x[31] + landmarks_x[35]) / 3.0, \
                                       (landmarks_y[33] + landmarks_y[31] + landmarks_y[35]) / 3.0

        return (nose_center_x, nose_center_y), \
               (left_eye_center_x, left_eye_center_y), (right_eye_center_x, right_eye_center_y)
