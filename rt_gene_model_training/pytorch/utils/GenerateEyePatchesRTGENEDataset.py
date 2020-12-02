from __future__ import print_function, division, absolute_import

import argparse
import os

import cv2
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase

script_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('im_path', type=str, default=os.path.join(script_path, '../samples/natural'),
                        nargs='?', help='Path to an image or a directory containing images')
    parser.add_argument('--output_path', type=str, default=os.path.join(script_path, '../samples/'), help='Output directory for left/right eye patches')

    landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                            checkpoint_path_face=os.path.join(script_path, "../../rt_gene/model_nets/SFD/s3fd_facedetector.pth"),
                                            checkpoint_path_landmark=os.path.join(script_path, "../../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar"),
                                            model_points_file=os.path.join(script_path, "../../rt_gene/model_nets/face_model_68.txt"))

    args = parser.parse_args()

    image_path_list = []
    if os.path.isfile(args.im_path):
        image_path_list.append(os.path.split(args.im_path)[1])
        args.im_path = os.path.split(args.im_path)[0]
    elif os.path.isdir(args.im_path):
        for image_file_name in os.listdir(args.im_path):
            if image_file_name.endswith('.jpg') or image_file_name.endswith('.png'):
                if '_gaze' not in image_file_name and '_headpose' not in image_file_name:
                    image_path_list.append(image_file_name)

    left_folder_path = os.path.join(args.output_path, "left_new")
    right_folder_path = os.path.join(args.output_path, "right_new")
    if not os.path.isdir(left_folder_path):
        os.makedirs(left_folder_path)
    if not os.path.isdir(right_folder_path):
        os.makedirs(right_folder_path)

    p_bar = tqdm(image_path_list)
    for image_file_name in p_bar:
        p_bar.set_description("Processing {}".format(image_file_name))
        image = cv2.imread(os.path.join(args.im_path, image_file_name))
        if image is None:
            continue

        faceboxes = landmark_estimator.get_face_bb(image)
        if len(faceboxes) == 0:
            continue

        subjects = landmark_estimator.get_subjects_from_faceboxes(image, faceboxes)
        for subject in subjects:
            le_c, re_c, _, _ = subject.get_eye_image_from_landmarks(subject, landmark_estimator.eye_image_size)

            if le_c is not None and re_c is not None:
                img_name = image_file_name.split(".")[0]
                left_image_path = ["left", img_name, "rgb.png"]
                left_image_path = os.path.join(left_folder_path, "_".join(left_image_path))

                right_image_path = ["right", img_name, "rgb.png"]
                right_image_path = os.path.join(right_folder_path, "_".join(right_image_path))

                cv2.imwrite(left_image_path, le_c)
                cv2.imwrite(right_image_path, re_c)
