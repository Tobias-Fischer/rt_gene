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
import os
import tensorflow
import rospkg
import scipy

import time
import rospkg
import rospy
import os
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image as pilImage

from rt_gene.Hopenet import Hopenet

import numpy as np

import rt_gene.gaze_tools as gaze_tools

from rt_gene.kalman_stabilizer import Stabilizer

from rt_gene.msg import MSG_SubjectImagesList
from rt_gene.subject_ros_bridge import SubjectListBridge


class SubjectDetected(object):
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

        self.CNN_INPUT_SIZE = 128

        self.model_points = self._get_full_model_points()

        self.sess_bb = None
        self.use_mtcnn = rospy.get_param("~use_mtcnn", True)
        if self.use_mtcnn:
            self.minsize = 40  # minimum size of face
            self.threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            self.factor = 0.709  # scale factor
            self.pnet, self.rnet, self.onet = self.load_mtcnn_network()
        else:
            rospy.logwarn(
                "This doesn't work with the currently shipped version of ROS_OpenCV (3.3.1-dev) but requires OpenCV-3.3.1")
            dnn_proto_txt = os.path.join(rospkg.RosPack().get_path('rt_gene'), 'model_nets/dnn_deploy.prototxt')
            dnn_model = os.path.join(rospkg.RosPack().get_path('rt_gene'),
                                     'model_nets/res10_300x300_ssd_iter_140000.caffemodel')
            self.face_net = cv2.dnn.readNetFromCaffe(dnn_proto_txt, dnn_model)

        mark_model = os.path.join(rospkg.RosPack().get_path('rt_gene'), 'model_nets/frozen_inference_graph.pb')

        detection_graph = tensorflow.Graph()
        with detection_graph.as_default():
            od_graph_def = tensorflow.GraphDef()
            with tensorflow.gfile.GFile(mark_model, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tensorflow.import_graph_def(od_graph_def, name='')
        self.graph = detection_graph

        gpu_options = tensorflow.GPUOptions(allow_growth=True, visible_device_list="0",
                                            per_process_gpu_memory_fraction=0.3)
        self.sess = tensorflow.Session(graph=detection_graph, config=tensorflow.ConfigProto(gpu_options=gpu_options,
                                                                                            log_device_placement=False,
                                                                                            inter_op_parallelism_threads=1,
                                                                                            intra_op_parallelism_threads=1
                                                                                            ))
        self.color_sub = rospy.Subscriber("/image", Image, self.callback, buff_size=2 ** 24, queue_size=1)

        hopenet_snapshot_path = os.path.join(rospkg.RosPack().get_path('rt_gene'),
                                     'model_nets/hopenet_robust_alpha1.pkl')

        # ResNet50 structure
        self.hopenet = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)

        print('Loading Hopenet Snapshot.')
        # Load snapshot
        saved_state_dict = torch.load(hopenet_snapshot_path)
        self.hopenet.load_state_dict(saved_state_dict)

        print('Loading data.')
        idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(idx_tensor).cuda(0)

        self.transformations = transforms.Compose([transforms.Resize(224),
                                                   transforms.CenterCrop(224), transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                        std=[0.229, 0.224, 0.225])])

        self.hopenet.cuda(0)
        self.hopenet.eval()

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

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
        # empirically, model_points * 1.1 works well with an IP of 0.058.
        model_points = model_points * (self.interpupillary_distance * 18.9)

        return model_points

    def get_face_bb_mtcnn(self, image, _):
        import rt_gene.detect_face as detect_face
        bounding_boxes, landmark_points = detect_face.detect_face(image, self.minsize, self.pnet, self.rnet, self.onet,
                                                                  self.threshold, self.factor)

        faceboxes = []
        for box in bounding_boxes[:, 0:4].astype(np.uint32):
            # Move box down.
            diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
            offset_y = int(abs(diff_height_width / 2))
            box_moved = self.move_box(box, [0, offset_y])

            # Make box square.
            facebox = self.get_square_box(box_moved)

            if self.box_in_image(facebox, image):
                faceboxes.append(facebox)
        return faceboxes

    # noinspection PyUnusedLocal
    def get_face_bb(self, image, timestamp):
        rows, cols, _ = image.shape
        threshold = 0.5

        faceboxes = []

        resized = cv2.resize(image, (300, 300))
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        self.face_net.setInput(cv2.dnn.blobFromImage(
            resized, 1.0, (300, 300), (104.0, 177.0, 123.0), False, False))
        detections = self.face_net.forward()

        for result in detections[0, 0, :, :]:
            confidence = result[2]
            if confidence > threshold:
                x_left_bottom = int(result[3] * cols)
                y_left_bottom = int(result[4] * rows)
                x_right_top = int(result[5] * cols)
                y_right_top = int(result[6] * rows)

                box = [x_left_bottom, y_left_bottom, x_right_top, y_right_top]
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

        # Make sure box is always square.
        assert ((right_x - left_x) == (bottom_y - top_y)), 'Box is not square.'

        return [left_x, top_y, right_x, bottom_y]

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

            success, _, translation_vector = self.get_head_pose(color_img, subject.marks, subject_id)
            rotation_vector = self.get_head_pose_deep_net(color_img, subject)

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

    def get_head_pose_deep_net(self, frame, subject):

        img = frame[subject.face_bb[1]: subject.face_bb[3], subject.face_bb[0]: subject.face_bb[2]]

        img = pilImage.fromarray(img)

        # Transform
        img = self.transformations(img)
        img_shape = img.size()
        img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
        img = Variable(img).cuda(0)

        yaw, pitch, roll = self.hopenet(img)

        yaw_predicted = F.softmax(yaw, dim=1)
        pitch_predicted = F.softmax(pitch, dim=1)
        roll_predicted = F.softmax(roll, dim=1)
        # Get continuous predictions in degrees.
        yaw_predicted = torch.sum(yaw_predicted.data[0] * self.idx_tensor) * 3 - 99
        pitch_predicted = torch.sum(pitch_predicted.data[0] * self.idx_tensor) * 3 - 99
        roll_predicted = torch.sum(roll_predicted.data[0] * self.idx_tensor) * 3 - 99

        p = pitch_predicted * np.pi / 180
        y = -(yaw_predicted * np.pi / 180)
        r = roll_predicted * np.pi / 180

        quat = tf_transformations.quaternion_from_euler(r + np.pi, p + np.pi, y)

        return quat

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

        return success, rot_head, translation_vector

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

    def detect_marks(self, image_np):
        """Detect marks from image"""
        # Get result tensor by its name.
        logits_tensor = self.graph.get_tensor_by_name('logits/BiasAdd:0')

        # Actual detection.
        predictions = self.sess.run(logits_tensor, feed_dict={'input_image_tensor:0': image_np})

        # Convert predictions to landmarks.
        marks = np.array(predictions).flatten()
        marks = np.reshape(marks, (-1, 2))

        return marks

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
            face_img = color_img[facebox[1]: facebox[3], facebox[0]: facebox[2]]
            face_img = cv2.resize(face_img, (self.CNN_INPUT_SIZE, self.CNN_INPUT_SIZE))
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

            marks = self.detect_marks(face_img)

            marks_orig = np.array(marks, copy=True)
            marks_orig *= self.CNN_INPUT_SIZE

            # Convert the marks locations from local CNN to global image.
            marks *= (facebox[2] - facebox[0])
            marks[:, 0] += facebox[0]
            marks[:, 1] += facebox[1]

            eye_indices = np.array([36, 39, 42, 45])

            transformed_landmarks = marks_orig[eye_indices]

            return face_img, transformed_landmarks, marks
        except Exception:
            return None, None, None

    def detect_landmarks(self, color_img, timestamp):
        if self.use_mtcnn:
            faceboxes = self.get_face_bb_mtcnn(color_img, timestamp)
        else:
            faceboxes = self.get_face_bb(color_img, timestamp)

        self.__update_subjects(faceboxes)

        for subject in self.subjects.values():
            res = self.__detect_landmarks_one_box(subject.face_bb, color_img)
            subject.face_color = res[0]
            subject.transformed_landmarks = res[1]
            subject.marks = res[2]

    def __get_eye_image_one(self, transformed_landmarks, face_aligned_color):
        margin_ratio = 1.0

        try:
            # Get the width of the eye, and compute how big the margin should be according to the width
            lefteye_width = transformed_landmarks[3][0] - transformed_landmarks[2][0]
            righteye_width = transformed_landmarks[1][0] - transformed_landmarks[0][0]
            lefteye_margin, righteye_margin = lefteye_width * margin_ratio, righteye_width * margin_ratio

            # lefteye_center_x = transformed_landmarks[2][0] + lefteye_width / 2
            # righteye_center_x = transformed_landmarks[0][0] + righteye_width / 2
            lefteye_center_y = (transformed_landmarks[2][1] + transformed_landmarks[3][1]) / 2
            righteye_center_y = (transformed_landmarks[1][1] + transformed_landmarks[0][1]) / 2

            desired_ratio = self.eye_image_size[0] / self.eye_image_size[1] / 2

            # Now compute the bounding boxes
            # The left / right x-coordinates are computed as the landmark position plus/minus the margin
            # The bottom / top y-coordinates are computed according to the desired ratio, as the width of the image is known
            left_bb = np.zeros(4, dtype=np.int32)
            left_bb[0] = transformed_landmarks[2][0] - lefteye_margin / 2
            left_bb[1] = lefteye_center_y - (lefteye_width + lefteye_margin) * desired_ratio * 1.25
            left_bb[2] = transformed_landmarks[3][0] + lefteye_margin / 2
            left_bb[3] = lefteye_center_y + (lefteye_width + lefteye_margin) * desired_ratio * 1.25

            right_bb = np.zeros(4, dtype=np.int32)
            right_bb[0] = transformed_landmarks[0][0] - righteye_margin / 2
            right_bb[1] = righteye_center_y - (righteye_width + righteye_margin) * desired_ratio * 1.25
            right_bb[2] = transformed_landmarks[1][0] + righteye_margin / 2
            right_bb[3] = righteye_center_y + (righteye_width + righteye_margin) * desired_ratio * 1.25

            # Extract the eye images from the aligned image
            left_eye_color = face_aligned_color[left_bb[1]:left_bb[3], left_bb[0]:left_bb[2], :]
            right_eye_color = face_aligned_color[right_bb[1]:right_bb[3], right_bb[0]:right_bb[2], :]

            # for p in transformed_landmarks:  # For debug visualization only
            #     cv2.circle(face_aligned_color, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

            # So far, we have only ensured that the ratio is correct. Now, resize it to the desired size.
            left_eye_color_resized = scipy.misc.imresize(left_eye_color, self.eye_image_size, interp='bilinear')
            right_eye_color_resized = scipy.misc.imresize(right_eye_color, self.eye_image_size, interp='bilinear')
        except (ValueError, TypeError):
            return None, None, None, None
        return left_eye_color_resized, right_eye_color_resized, left_bb, right_bb

    # noinspection PyUnusedLocal
    def get_eye_image(self):
        """Extract the left and right eye images given the (dlib) transformed_landmarks and the source image.
        First, align the face. Then, extract the width of the eyes given the landmark positions.
        The height of the images is computed according to the desired ratio of the eye images."""

        start_time = time.time()
        for subject in self.subjects.values():
            res = self.__get_eye_image_one(subject.transformed_landmarks, subject.face_color)
            subject.left_eye_color = res[0]
            subject.right_eye_color = res[1]
            subject.left_eye_bb = res[2]
            subject.right_eye_bb = res[3]

        tqdm.write('New get_eye_image time: ' + str(time.time() - start_time))

    @staticmethod
    def get_image_points_headpose(landmarks):
        return landmarks

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

    def load_mtcnn_network(self):
        import rt_gene.detect_face as detect_face
        import tensorflow

        """Load the MTCNN network."""
        tqdm.write('Creating networks and loading parameters')

        with tensorflow.Graph().as_default():
            gpu_memory_fraction = rospy.get_param("~gpu_memory_fraction", 0.05)
            gpu_options = tensorflow.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
            # gpu_options = tensorflow.GPUOptions(allow_growth=True, visible_device_list="0")
            self.sess_bb = tensorflow.Session(config=tensorflow.ConfigProto(gpu_options=gpu_options,
                                                                            log_device_placement=False))
            with self.sess_bb.as_default():
                model_path = rospkg.RosPack().get_path('rt_gene') + '/model_nets'
                pnet, rnet, onet = detect_face.create_mtcnn(self.sess_bb, model_path)
        return pnet, rnet, onet

