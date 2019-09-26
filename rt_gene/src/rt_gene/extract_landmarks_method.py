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
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from tqdm import tqdm
from image_geometry import PinholeCameraModel
import tf.transformations as tf_transformations
from geometry_msgs.msg import PointStamped, Point
from tf import TransformBroadcaster, TransformListener, ExtrapolationException
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
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
from rt_gene.tracker_face_encoding import FaceEncodingTracker
from rt_gene.tracker_sequential import SequentialTracker
from rt_gene.tracker_generic import TrackedSubject

from face_alignment.detection.sfd import FaceDetector
from rt_gene.ThreeDDFA.inference import predict_68pts, crop_img, parse_roi_box_from_bbox, parse_roi_box_from_landmark
from rt_gene.ThreeDDFA.ddfa import ToTensorGjz, NormalizeGjz


facial_landmark_transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])


class LandmarkMethod(object):
    def __init__(self, img_proc=None):
        self.subject_tracker = FaceEncodingTracker() if rospy.get_param("~use_face_encoding_tracker",
                                                                        default=False) else SequentialTracker()
        self.bridge = CvBridge()
        self.__subject_bridge = SubjectListBridge()
        self.model_size_rescale = 30.0
        self.head_pitch = 0.0
        self.interpupillary_distance = 0.058

        self.device_id_facedetection = rospy.get_param("~device_id_facedetection", default="cuda:0")
        self.device_id_facealignment = rospy.get_param("~device_id_facealignment", default="cuda:0")
        rospy.loginfo("Using device {} for face detection.".format(self.device_id_facedetection))
        rospy.loginfo("Using device {} for face alignment.".format(self.device_id_facealignment))

        self.rgb_frame_id = rospy.get_param("~rgb_frame_id", "/kinect2_link")
        self.rgb_frame_id_ros = rospy.get_param("~rgb_frame_id_ros", "/kinect2_nonrotated_link")

        self.model_points = None
        self.eye_image_size = (rospy.get_param("~eye_image_width", 60), rospy.get_param("~eye_image_height", 36))

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

        self.model_points = LandmarkMethod.get_full_model_points(self.interpupillary_distance, self.model_size_rescale)

        self.face_net = FaceDetector(device=self.device_id_facedetection)

        self.facial_landmark_nn = LandmarkMethod.load_face_landmark_model()

        Server(ModelSizeConfig, self._dyn_reconfig_callback)

        # have the subscriber the last thing that's run to avoid conflicts
        self.color_sub = rospy.Subscriber("/image", Image, self.process_image, buff_size=2 ** 24, queue_size=1)

    @staticmethod
    def load_face_landmark_model():
        import rt_gene.ThreeDDFA.mobilenet_v1 as mobilenet_v1
        checkpoint_fp = rospkg.RosPack().get_path('rt_gene') + '/model_nets/phase1_wpdc_vdc.pth.tar'
        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        cudnn.benchmark = True
        model = model.cuda()
        model.eval()
        return model

    def _dyn_reconfig_callback(self, config, _):
        self.model_points /= (self.model_size_rescale * self.interpupillary_distance)
        self.model_size_rescale = config["model_size"]
        self.interpupillary_distance = config["interpupillary_distance"]
        self.model_points *= (self.model_size_rescale * self.interpupillary_distance)
        self.head_pitch = config["head_pitch"]
        return config

    @staticmethod
    def get_full_model_points(interpupillary_distance, model_size_rescale):
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
        model_points = model_points * (interpupillary_distance * model_size_rescale)

        return model_points

    @staticmethod
    def get_face_bb(face_net, image):
        faceboxes = []
        start_time = time.time()
        fraction = 4.0
        image = cv2.resize(image, (0, 0), fx=1.0 / fraction, fy=1.0 / fraction)
        detections = face_net.detect_from_image(image)
        tqdm.write(
            "Face Detector Frequency: {:.2f}Hz for {} Faces".format(1 / (time.time() - start_time), len(detections)))

        for result in detections:
            # scale back up to image size
            box = result[:4]
            confidence = result[4]

            if gaze_tools.box_in_image(box, image) and confidence > 0.8:
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = gaze_tools.move_box(box, [0, offset_y])

                # Make box square.
                facebox = gaze_tools.get_square_box(box_moved)
                faceboxes.append(facebox)

        return faceboxes

    def process_image(self, color_msg):
        tqdm.write('Time now: {} message color: {} diff: {:.2f}s'.format((rospy.Time.now().to_sec()), color_msg.header.stamp.to_sec(), rospy.Time.now().to_sec() - color_msg.header.stamp.to_sec()))

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

            success, rotation_quaternion, translation_vector = self.get_head_pose(subject.marks, subject_id)

            # Publish all the data
            translation_headpose_tf = self.get_head_translation(timestamp, subject_id)

            if success:
                if translation_headpose_tf is not None:
                    roll_pitch_yaw = self.publish_pose(timestamp, translation_headpose_tf, rotation_quaternion, subject_id)

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
            if self.last_rvec[subject_id] is not None and self.last_tvec[subject_id] is not None and self.use_previous_headpose_estimate:
                (success, rotation_vector_unstable, translation_vector_unstable) = \
                    cv2.solvePnP(self.model_points, image_points_headpose, camera_matrix, dist_coeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE, useExtrinsicGuess=True,
                                 rvec=self.last_rvec[subject_id], tvec=self.last_tvec[subject_id])
            else:
                (success, rotation_vector_unstable, translation_vector_unstable) = \
                    cv2.solvePnP(self.model_points, image_points_headpose, camera_matrix, dist_coeffs,
                                 flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error as e:
            print('Could not estimate head pose', e)
            return False, None, None

        if not success:
            print('Could not estimate head pose')
            return False, None, None

        rotation_vector, translation_vector = self.apply_kalman_filter_head_pose(subject_id, rotation_vector_unstable, translation_vector_unstable)

        if not gaze_tools.is_rotation_vector_stable(self.last_rvec[subject_id], rotation_vector):
            return False, None, None

        self.last_rvec[subject_id] = rotation_vector
        self.last_tvec[subject_id] = translation_vector

        rotation_vector_swapped = [-rotation_vector[2], -rotation_vector[1] + np.pi + self.head_pitch,
                                   rotation_vector[0]]
        rot_head = tf_transformations.quaternion_from_euler(*rotation_vector_swapped)

        return success, rot_head, translation_vector

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

    @staticmethod
    def visualize_headpose_result(face_image, est_headpose):
        """Here, we take the original eye eye_image and overlay the estimated headpose."""
        output_image = np.copy(face_image)

        center_x = output_image.shape[1] / 2
        center_y = output_image.shape[0] / 2

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

    def publish_subject_list(self, timestamp, subjects):
        assert (subjects is not None)

        subject_list_message = self.__subject_bridge.images_to_msg(subjects, timestamp)

        self.subject_pub.publish(subject_list_message)

    def publish_pose(self, timestamp, nose_center_3d_tf, rot_head, subject_id):
        self.tf_broadcaster.sendTransform(nose_center_3d_tf,
                                          rot_head,
                                          timestamp,
                                          self.tf_prefix + "/head_pose_estimated" + str(subject_id),
                                          self.rgb_frame_id_ros)

        return gaze_tools.limit_yaw(rot_head)

    @staticmethod
    def transform_landmarks(landmarks, box):
        eye_indices = np.array([36, 39, 42, 45])
        transformed_landmarks = landmarks[eye_indices]
        transformed_landmarks[:, 0] -= box[0]
        transformed_landmarks[:, 1] -= box[1]
        return transformed_landmarks

    @staticmethod
    def ddfa_forward_pass(facial_landmark_nn, color_img, roi_box):
        img_step = crop_img(color_img, roi_box)
        img_step = cv2.resize(img_step, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
        _input = facial_landmark_transform(img_step).unsqueeze(0)
        with torch.no_grad():
            _input = _input.cuda()
            param = facial_landmark_nn(_input)
            param = param.squeeze().cpu().numpy().flatten().astype(np.float32)

        return predict_68pts(param, roi_box)

    def update_subject_tracker(self, color_img):
        faceboxes = LandmarkMethod.get_face_bb(self.face_net, color_img)
        if len(faceboxes) == 0:
            self.subject_tracker.clear_elements()
            return

        face_images = [gaze_tools.crop_face_from_image(color_img, b) for b in faceboxes]

        tracked_subjects = []
        for facebox, face_image in zip(faceboxes, face_images):
            roi_box = parse_roi_box_from_bbox(facebox)
            initial_pts68 = self.ddfa_forward_pass(self.facial_landmark_nn, color_img, roi_box)
            roi_box_refined = parse_roi_box_from_landmark(initial_pts68)
            pts68 = self.ddfa_forward_pass(self.facial_landmark_nn, color_img, roi_box_refined)

            np_landmarks = np.array((pts68[0], pts68[1])).T
            transformed_landmarks = LandmarkMethod.transform_landmarks(np_landmarks, facebox)
            subject = TrackedSubject(np.array(facebox), face_image, transformed_landmarks, np_landmarks)
            tracked_subjects.append(subject)

        # track the new faceboxes according to the previous ones
        self.subject_tracker.track(tracked_subjects)
