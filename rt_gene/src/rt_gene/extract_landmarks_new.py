#!/usr/bin/env python

"""
Extract landmarks using https://github.com/yinguobing/head-pose-estimation
Under MIT license: https://opensource.org/licenses/MIT
@Tobias Fischer (t.fischer@imperial.ac.uk)
@Ahmed Al-Hindawi <a.al-hindawi@imperial.ac.uk>
"""

from __future__ import print_function, division, absolute_import

import os
import rospkg
import rospy
import time
import numpy as np
import scipy.misc
from tqdm import tqdm
from sensor_msgs.msg import Image
import cv2
import tensorflow
from math import sqrt

# noinspection PyUnresolvedReferences
from rt_gene.extract_landmarks_method import LandmarkMethod, SubjectDetected

# noinspection PyUnresolvedReferences
import rt_gene.gaze_tools as gaze_tools


class LandmarkNew(LandmarkMethod):
    def __init__(self):
        super(LandmarkNew, self).__init__()

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
                distance_matrix[i][j] = sqrt(np.mean(((np.array(subject.face_bb) - np.array(b_new)) ** 2)))
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
