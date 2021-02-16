#!/usr/bin/env python

# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

from __future__ import print_function, division, absolute_import

import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw
from rt_gene.gaze_tools_standalone import euler_from_matrix

script_path = os.path.dirname(os.path.realpath(__file__))


def load_camera_calibration(calibration_file):
    import yaml
    with open(calibration_file, 'r') as f:
        cal = yaml.safe_load(f)

    dist_coefficients = np.array(cal['distortion_coefficients']['data'], dtype='float32').reshape(1, 5)
    camera_matrix = np.array(cal['camera_matrix']['data'], dtype='float32').reshape(3, 3)

    return dist_coefficients, camera_matrix


def extract_eye_image_patches(subjects):
    for subject in subjects:
        le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c


def estimate_gaze(base_name, color_img, dist_coefficients, camera_matrix):
    faceboxes = landmark_estimator.get_face_bb(color_img)
    if len(faceboxes) == 0:
        tqdm.write('Could not find faces in the image')
        return

    subjects = landmark_estimator.get_subjects_from_faceboxes(color_img, faceboxes)
    extract_eye_image_patches(subjects)

    input_r_list = []
    input_l_list = []
    input_head_list = []
    valid_subject_list = []

    for idx, subject in enumerate(subjects):
        if subject.left_eye_color is None or subject.right_eye_color is None:
            tqdm.write('Failed to extract eye image patches')
            continue

        success, rotation_vector, _ = cv2.solvePnP(landmark_estimator.model_points,
                                                   subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                   cameraMatrix=camera_matrix,
                                                   distCoeffs=dist_coefficients, flags=cv2.SOLVEPNP_DLS)

        if not success:
            tqdm.write('Not able to extract head pose for subject {}'.format(idx))
            continue

        _rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        _rotation_matrix = np.matmul(_rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        _m = np.zeros((4, 4))
        _m[:3, :3] = _rotation_matrix
        _m[3, 3] = 1
        # Go from camera space to ROS space
        _camera_to_ros = [[0.0, 0.0, 1.0, 0.0],
                          [-1.0, 0.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0, 0.0],
                          [0.0, 0.0, 0.0, 1.0]]
        roll_pitch_yaw = list(euler_from_matrix(np.dot(_camera_to_ros, _m)))
        roll_pitch_yaw = limit_yaw(roll_pitch_yaw)

        phi_head, theta_head = get_phi_theta_from_euler(roll_pitch_yaw)

        face_image_resized = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
        head_pose_image = landmark_estimator.visualize_headpose_result(face_image_resized, (phi_head, theta_head))

        if args.vis_headpose:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(head_pose_image, cv2.COLOR_BGR2RGB))
            plt.show()

        if args.save_headpose:
            # add idx to cope with multiple persons in one image
            cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_headpose_%s.jpg'%(idx)), head_pose_image)

        input_r_list.append(gaze_estimator.input_from_image(subject.right_eye_color))
        input_l_list.append(gaze_estimator.input_from_image(subject.left_eye_color))
        input_head_list.append([theta_head, phi_head])
        valid_subject_list.append(idx)

    if len(valid_subject_list) == 0:
        return

    gaze_est = gaze_estimator.estimate_gaze_twoeyes(inference_input_left_list=input_l_list,
                                                    inference_input_right_list=input_r_list,
                                                    inference_headpose_list=input_head_list)

    for subject_id, gaze, headpose in zip(valid_subject_list, gaze_est.tolist(), input_head_list):
        subject = subjects[subject_id]
        # Build visualizations
        r_gaze_img = gaze_estimator.visualize_eye_result(subject.right_eye_color, gaze)
        l_gaze_img = gaze_estimator.visualize_eye_result(subject.left_eye_color, gaze)
        s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

        if args.vis_gaze:
            plt.axis("off")
            plt.imshow(cv2.cvtColor(s_gaze_img, cv2.COLOR_BGR2RGB))
            plt.show()

        if args.save_gaze:
            # add subject_id to cope with multiple persons in one image
            cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_gaze_%s.jpg'%(subject_id)), s_gaze_img)
            # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_left.jpg'), subject.left_eye_color)
            # cv2.imwrite(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_right.jpg'), subject.right_eye_color)

        if args.save_estimate:
            # add subject_id to cope with multiple persons in one image
            with open(os.path.join(args.output_path, os.path.splitext(base_name)[0] + '_output_%s.txt'%(subject_id)), 'w+') as f:
                f.write(os.path.splitext(base_name)[0] + ', [' + str(headpose[1]) + ', ' + str(headpose[0]) + ']' +
                        ', [' + str(gaze[1]) + ', ' + str(gaze[0]) + ']' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/')),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--calib-file', type=str, dest='calib_file', default=None, help='Camera calibration file')
    parser.add_argument('--vis-headpose', dest='vis_headpose', action='store_true', help='Display the head pose images')
    parser.add_argument('--no-vis-headpose', dest='vis_headpose', action='store_false', help='Do not display the head pose images')
    parser.add_argument('--save-headpose', dest='save_headpose', action='store_true', help='Save the head pose images')
    parser.add_argument('--no-save-headpose', dest='save_headpose', action='store_false', help='Do not save the head pose images')
    parser.add_argument('--vis-gaze', dest='vis_gaze', action='store_true', help='Display the gaze images')
    parser.add_argument('--no-vis-gaze', dest='vis_gaze', action='store_false', help='Do not display the gaze images')
    parser.add_argument('--save-gaze', dest='save_gaze', action='store_true', help='Save the gaze images')
    parser.add_argument('--save-estimate', dest='save_estimate', action='store_true', help='Save the predictions in a text file')
    parser.add_argument('--no-save-gaze', dest='save_gaze', action='store_false', help='Do not save the gaze images')
    parser.add_argument('--gaze_backend', choices=['tensorflow', 'pytorch'], default='tensorflow')
    parser.add_argument('--output_path', type=str, default=os.path.abspath(os.path.join(script_path, './samples_gaze/out')),
                        help='Output directory for head pose and gaze images')
    parser.add_argument('--models', nargs='+', type=str, default=[os.path.abspath(os.path.join(script_path, '../rt_gene/model_nets/Model_allsubjects1.h5'))],
                        help='List of gaze estimators')
    parser.add_argument('--device-id-facedetection', dest="device_id_facedetection", type=str, default='cuda:0', help='Pytorch device id. Set to "cpu:0" to disable cuda')

    parser.set_defaults(vis_gaze=True)
    parser.set_defaults(save_gaze=True)
    parser.set_defaults(vis_headpose=False)
    parser.set_defaults(save_headpose=True)
    parser.set_defaults(save_estimate=False)

    args = parser.parse_args()

    image_path_list = []
    if os.path.isfile(args.im_path):
        image_path_list.append(os.path.split(args.im_path)[1])
        args.im_path = os.path.split(args.im_path)[0]
    elif os.path.isdir(args.im_path):
        for image_file_name in sorted(os.listdir(args.im_path)):
            if image_file_name.lower().endswith('.jpg') or image_file_name.lower().endswith('.png') or image_file_name.lower().endswith('.jpeg'):
                if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
                    image_path_list.append(image_file_name)
    else:
        tqdm.write('Provide either a path to an image or a path to a directory containing images')
        sys.exit(1)

    tqdm.write('Loading networks')
    landmark_estimator = LandmarkMethodBase(device_id_facedetection=args.device_id_facedetection,
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../rt_gene/model_nets/face_model_68.txt")))

    if args.gaze_backend == "tensorflow":
        from rt_gene.estimate_gaze_tensorflow import GazeEstimator

        gaze_estimator = GazeEstimator("/gpu:0", args.models)
    elif args.gaze_backend == "pytorch":
        from rt_gene.estimate_gaze_pytorch import GazeEstimator

        gaze_estimator = GazeEstimator("cuda:0", args.models)
    else:
        raise ValueError("Incorrect gaze_base backend, choices are: tensorflow or pytorch")

    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)

    for image_file_name in tqdm(image_path_list):
        tqdm.write('Estimate gaze on ' + image_file_name)
        image = cv2.imread(os.path.join(args.im_path, image_file_name))
        if image is None:
            tqdm.write('Could not load ' + image_file_name + ', skipping this image.')
            continue

        if args.calib_file is not None:
            _dist_coefficients, _camera_matrix = load_camera_calibration(args.calib_file)
        else:
            im_width, im_height = image.shape[1], image.shape[0]
            tqdm.write('WARNING!!! You should provide the camera calibration file, otherwise you might get bad results. Using a crude approximation!')
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

        estimate_gaze(image_file_name, image, _dist_coefficients, _camera_matrix)
