#!/usr/bin/env python

import sys
import os
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from rt_gene.gaze_tools import get_phi_theta_from_euler
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.estimate_gaze_base import GazeEstimatorBase


script_path = os.path.dirname(os.path.realpath(__file__))


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix


def rotation_vector_to_rpy(rotation_vector):
    roll_pitch_yaw = [-rotation_vector[2], rotation_vector[1], rotation_vector[0]]
    roll_pitch_yaw[0] += np.pi
    if roll_pitch_yaw[0] > np.pi:
        roll_pitch_yaw[0] -= 2 * np.pi
    return roll_pitch_yaw


def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, le_bb, re_bb = subject.get_eye_image_from_landmarks(subject.transformed_landmarks, subject.face_color, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c
        subject.left_eye_bb = le_bb
        subject.right_eye_bb = re_bb


def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix):
    faceboxes = landmark_estimator.get_face_bb(color_img)
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

    for idx, subject in enumerate(subjects):
        (success, rotation_vector, translation_vector) = cv2.solvePnP(landmark_estimator.model_points, subject.marks, cameraMatrix=camera_matrix,
                                                                      distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_ITERATIVE,
                                                                      useExtrinsicGuess=True,
                                                                      rvec=landmark_estimator.rvec_init.copy(),
                                                                      tvec=landmark_estimator.tvec_init.copy())
        roll_pitch_yaw = rotation_vector_to_rpy(rotation_vector.flatten().tolist())

        if roll_pitch_yaw is not None:
            phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

            face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
            head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

            if args.vis_headpose:
                plt.axis("off")
                plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
                plt.show()
        else:
            tqdm.write('Not able to extract head pose for subject {}'.format(idx))
            continue

        input_r = gaze_estimator.input_from_image(subject.right_eye_color)
        input_l = gaze_estimator.input_from_image(subject.left_eye_color)
        gaze_est = gaze_estimator.estimate_gaze_twoeyes([input_l], [input_r], [[theta_head, phi_head]])[0]

        # Build visualizations
        r_gaze_img = gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze_est)
        l_gaze_img = gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze_est)
        s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        if args.vis_gaze:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
            plt.show()

        cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_headpose.jpg'), head_pose_image)
        cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_gaze.jpg'), s_gaze_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.join(script_path, '../samples/'),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--output_path', type=str, default=os.path.join(script_path, '../samples/out'), help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.join(script_path, '../model_nets/Model_allsubjects1.h5')], help='List of gaze estimators')

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

    tqdm.write('Loading networks')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                            checkpoint_path_landmark=os.path.join(script_path, "../model_nets/phase1_wpdc_vdc.pth.tar"),
                                            model_points_file=os.path.join(script_path, "../model_nets/face_model_68.txt"))
    gaze_estimator = GazeEstimatorBase(device_id_gaze="/gpu:0", model_files=args.models)

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    for image_filename in image_path_list:
        image = cv2.imread(os.path.join(args.im_path, image_filename))
        if image is None:
            tqdm.write('Could not load ' + image_filename + ', skipping this image.')

        if args.calib_file is not None:
            _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
        else:
            im_width, im_height = image.shape[1], image.shape[0]
            tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array([[im_height / 1.05, 0.0, im_width / 2.0], [0.0, im_height / 1.05, im_height / 2.0], [0.0, 0.0, 1.0]])

        estimate_gaze(image_filename, image, _dist_coefficients, _camera_matrix)
