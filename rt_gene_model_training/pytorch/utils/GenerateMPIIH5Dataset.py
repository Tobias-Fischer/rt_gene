import argparse
import os
from glob import glob
from math import asin, atan2

import cv2
import h5py
import numpy as np
import scipy.io as sio
from PIL import ImageFilter, ImageOps
from torchvision import transforms
from tqdm import tqdm

_required_size = (224, 224)
_transforms_list = [transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),  # equivalent to random 5px from each edge
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),
                    transforms.Grayscale(num_output_channels=3),
                    lambda x: x.filter(ImageFilter.GaussianBlur(radius=1)),
                    lambda x: x.filter(ImageFilter.GaussianBlur(radius=3)),
                    lambda x: ImageOps.equalize(x)]  # histogram equalisation


def transform_and_augment(image, augment=False):
    augmented_images = [np.array(trans(image)) for trans in _transforms_list if augment is True]
    augmented_images.append(np.array(image))

    return np.array(augmented_images, dtype=np.uint8)


def normalize_img(img, target_3d, head_rotation, gc, roi_size, cam_matrix, focal_new=960, distance_new=600):
    if roi_size is None:
        roi_size = (60, 36)

    distance = np.linalg.norm(target_3d)
    z_scale = distance_new / distance
    cam_new = np.array([[focal_new, 0, roi_size[0] / 2],
                        [0.0, focal_new, roi_size[1] / 2],
                        [0, 0, 1.0]])
    scale_mat = np.array([[1.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0],
                          [0.0, 0.0, z_scale]])
    h_rx = head_rotation[:, 0]
    forward = (target_3d / distance)
    down = np.cross(forward, h_rx)
    down = down / np.linalg.norm(down)
    right = np.cross(down, forward)
    right = right / np.linalg.norm(right)

    rot_mat = np.array([right.T, down.T, forward.T])
    warp_mat = (cam_new @ scale_mat) @ (rot_mat @ np.linalg.inv(cam_matrix))
    img_warped = cv2.warpPerspective(img, warp_mat, roi_size)

    # rotation normalization
    cnv_mat = scale_mat @ rot_mat
    h_rnew = cnv_mat @ head_rotation
    hrnew = cv2.Rodrigues(h_rnew)[0].reshape((3,))
    htnew = cnv_mat @ target_3d

    # gaze vector normalization
    gcnew = cnv_mat @ gc
    gvnew = gcnew - htnew
    gvnew = gvnew / np.linalg.norm(gvnew)

    return img_warped, hrnew, gvnew


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('--mpii_root', type=str, required=True, nargs='?', help='Path to the base directory of MPII')
    parser.add_argument('--augment_dataset', type=bool, required=False, default=False, help="Whether to augment the dataset with predefined transforms")
    parser.add_argument('--compress', action='store_true', dest="compress")
    parser.add_argument('--no-compress', action='store_false', dest="compress")
    parser.set_defaults(compress=False)
    args = parser.parse_args()

    _compression = "lzf" if args.compress is True else None

    hdf_file = h5py.File(os.path.abspath(os.path.join(args.mpii_root, 'mpii_dataset.hdf5')), mode='w')
    faceModel = sio.loadmat(os.path.join(args.mpii_root, '6 points-based face model.mat'))["model"]
    subjects = [os.path.join(args.mpii_root, "Data", "Original", 'p{:02d}/'.format(_i)) for _i in range(0, 15)]

    for subject_id, subject_path in enumerate(subjects):
        data_files = sorted(glob(os.path.join(subject_path, "*.mat")))
        subject_id = str("s{:03d}".format(subject_id))
        subject_grp = hdf_file.create_group(subject_id)
        data_store_idx = 0
        camera_calibration = os.path.join(subject_path, "Calibration/Camera.mat")
        camera_matrix = sio.loadmat(camera_calibration)["cameraMatrix"]
        days = sorted(list(glob(os.path.join(subject_path, "day*"))))
        for day in tqdm(days, desc="Subject {}".format(subject_id)):
            with open(os.path.join(day, "annotation.txt"), "r") as reader:
                image_annotations = reader.readlines()

            for idx, annotation in enumerate(image_annotations):
                annotations = [float(num) for num in annotation.split(" ")]
                image_name = "{:0=6d}".format(data_store_idx)
                image_grp = subject_grp.create_group(image_name)
                data_store_idx += 1

                img = cv2.imread(os.path.join(day, str("{:04d}.jpg".format(idx + 1))))
                headpose_hr = np.array(annotations[29:32])
                headpose_ht = np.array(annotations[32:35])
                hR = cv2.Rodrigues(headpose_hr)[0]
                Fc = np.dot(hR, faceModel)
                Fc = headpose_ht.T[:, np.newaxis] + Fc

                right_eye_center = 0.5 * (Fc[:, 0] + Fc[:, 1])
                left_eye_center = 0.5 * (Fc[:, 2] + Fc[:, 3])

                gaze_target = np.array(annotations[26:29]).T

                right_image, right_headpose, right_gaze = normalize_img(img, right_eye_center, hR, gaze_target, (60, 36), camera_matrix)
                left_image, left_headpose, left_gaze = normalize_img(img, left_eye_center, hR, gaze_target, (60, 36), camera_matrix)

                left_eye_theta = asin(-1 * left_gaze[1])
                left_eye_phi = atan2(-1 * left_gaze[0], -1 * left_gaze[2])

                right_eye_theta = asin(-1 * right_gaze[1])
                right_eye_phi = atan2(-1 * right_gaze[0], -1 * right_gaze[2])

                gaze_theta = (left_eye_theta + right_eye_theta) / 2.0
                gaze_phi = (left_eye_phi + right_eye_phi) / 2.0

                left_rotation_matrix = cv2.Rodrigues(left_headpose)[0]  # ignore the Jackobian matrix
                left_zv = left_rotation_matrix[:, 2]
                left_head_theta = asin(left_zv[1])
                left_head_phi = atan2(left_zv[0], left_zv[2])

                right_rotation_matrix = cv2.Rodrigues(right_headpose)[0]  # ignore the Jackobian matrix
                right_zv = right_rotation_matrix[:, 2]
                right_head_theta = asin(right_zv[1])
                right_head_phi = atan2(right_zv[0], right_zv[2])

                head_theta = (left_head_theta + right_head_theta) / 2.0
                head_phi = (left_head_phi + right_head_phi) / 2.0

                labels = [(head_theta, head_phi), (gaze_theta, gaze_phi)]
                left_data = transform_and_augment(left_image, augment=args.augment_dataset)
                right_data = transform_and_augment(right_image, augment=args.augment_dataset)
                image_grp.create_dataset("left", data=left_data, compression=_compression)
                image_grp.create_dataset("right", data=right_data, compression=_compression)
                image_grp.create_dataset("label", data=labels)

    hdf_file.flush()
    hdf_file.close()

