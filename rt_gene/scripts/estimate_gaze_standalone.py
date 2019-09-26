#!/usr/bin/env python

import sys
import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow
from keras.backend.tensorflow_backend import set_session
from keras.models import load_model

from face_alignment.detection.sfd import FaceDetector

from rt_gene.gaze_tools import crop_face_from_image, accuracy_angle, angle_loss, get_phi_theta_from_euler
from rt_gene.ThreeDDFA.inference import parse_roi_box_from_bbox, parse_roi_box_from_landmark
from rt_gene.extract_landmarks_method import LandmarkMethod
from rt_gene.tracker_generic import TrackedSubject
from estimate_gaze import GazeEstimator


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix


def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix):
    print('Extracting face bounding boxes')
    faceboxes = LandmarkMethod.get_face_bb(face_net, color_img)
    if len(faceboxes) == 0:
        print('Could not find faces in the image')
        return
    face_images = [crop_face_from_image(color_img, b) for b in faceboxes]

    print('Extracting landmark positions')
    subjects = []
    for facebox, face_image in zip(faceboxes, face_images):
        roi_box = parse_roi_box_from_bbox(facebox)
        initial_pts68 = LandmarkMethod.ddfa_forward_pass(facial_landmark_nn, color_img, roi_box)
        roi_box_refined = parse_roi_box_from_landmark(initial_pts68)
        pts68 = LandmarkMethod.ddfa_forward_pass(facial_landmark_nn, color_img, roi_box_refined)

        np_landmarks = np.array((pts68[0], pts68[1])).T
        transformed_landmarks = LandmarkMethod.transform_landmarks(np_landmarks, facebox)
        subjects.append(TrackedSubject(np.array(facebox), face_image, transformed_landmarks, np_landmarks))
    print('Extracted landmarks for {} subject(s)'.format(len(subjects)))

    print('Extracting eye image patches')
    eye_image_size = (60, 36)

    for subject in subjects:
        le_c, re_c, le_bb, re_bb = subject.get_eye_image_from_landmarks(subject.transformed_landmarks, subject.face_color, eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c
        subject.left_eye_bb = le_bb
        subject.right_eye_bb = re_bb

    print('Getting head pose')
    rvec_init = np.array([[0.01891013], [0.08560084], [-3.14392813]])
    tvec_init = np.array([[-14.97821226], [-10.62040383], [-2053.03596872]])

    model_size_rescale = 30.0
    interpupillary_distance = 0.058
    model_points = LandmarkMethod.get_full_model_points(interpupillary_distance, model_size_rescale)

    for idx, subject in enumerate(subjects):
        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, subject.marks, cameraMatrix=camera_matrix,
                                                                      distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_ITERATIVE,
                                                                      useExtrinsicGuess=True, rvec=rvec_init, tvec=tvec_init)
        roll_pitch_yaw = np.array([-rotation_vector[2], rotation_vector[1], rotation_vector[0]])
        roll_pitch_yaw[0] += np.pi
        if roll_pitch_yaw[0] > np.pi:
            roll_pitch_yaw[0] -= 2 * np.pi

        if roll_pitch_yaw is not None:
            phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

            face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            head_pose_image = LandmarkMethod.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

            if args.vis_headpose:
                plt.axis("off")
                plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
                plt.show()
        else:
            print('Not able to extract head pose for subject {}'.format(idx))
            continue

        print('Estimating gaze')
        input_r = GazeEstimator.input_from_image(subject.right_eye_color)
        input_l = GazeEstimator.input_from_image(subject.left_eye_color)
        gaze_est = GazeEstimator.estimate_gaze_twoeyes_standalone(input_l, input_r,
                                                                  np.array([theta_head, phi_head]), graph, models)
        r_gaze_img = GazeEstimator.visualize_eye_result(subject.right_eye_color, gaze_est)
        l_gaze_img = GazeEstimator.visualize_eye_result(subject.left_eye_color, gaze_est)
        s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        if args.vis_gaze:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
            plt.show()

        cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_headpose.jpg'), head_pose_image)
        cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_gaze.jpg'), s_gaze_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default='../samples/', nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_true', help='Do not display the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_true', help='Do not display the gaze images')
    parser.add_argument('--output_path', type=str, default='../samples/out', help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=['../model_nets/Model_allsubjects1.h5'], help='List of gaze estimators')

    parser.set_defaults(vis_gaze=True)
    parser.set_defaults(vis_headpose=False)

    args = parser.parse_args()

    image_path_list = []
    if os.path.isfile(args.im_path):
        image_path_list.append(args.im_path)
    elif os.path.isdir(args.im_path):
        for image_file_name in os.listdir(args.im_path):
            if image_file_name.endswith('.jpg') or image_file_name.endswith('.png'):
                image_path_list.append(image_file_name)
    else:
        print('Provide either a path to an image or a path to a directory containing images')
        sys.exit(1)

    print('Loading networks')
    face_net = FaceDetector(device='cuda:0')
    facial_landmark_nn = LandmarkMethod.load_face_landmark_model()

    with tensorflow.device("/gpu:0"):
        config = tensorflow.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.log_device_placement = False
        sess = tensorflow.Session(config=config)
        set_session(sess)

    models = []
    for model_file in args.models:
        print('Load model ' + model_file)
        model = load_model(model_file, custom_objects={'accuracy_angle': accuracy_angle, 'angle_loss': angle_loss})
        # noinspection PyProtectedMember
        model._make_predict_function()  # have to initialize before threading
        models.append(model)

    graph = tensorflow.get_default_graph()

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    for image_filename in image_path_list:
        image = cv2.imread(os.path.join(args.im_path, image_filename))
        if image is None:
            print('Could not load ' + image_filename + ', skipping this image.')

        if args.calib_file is not None:
            _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
        else:
            im_width, im_height = image.shape[1], image.shape[0]
            print('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array([[im_height / 1.05, 0.0, im_width / 2.0], [0.0, im_height / 1.05, im_height / 2.0], [0.0, 0.0, 1.0]])

        estimate_gaze(image_filename, image, _dist_coefficients, _camera_matrix)
