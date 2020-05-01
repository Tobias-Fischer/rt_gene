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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('--mpii_root', type=str, required=True, nargs='?', help='Path to the base directory of MPII')
    parser.add_argument('--augment_dataset', type=bool, required=False, default=False, help="Whether to augment the dataset with predefined transforms")
    parser.add_argument('--compress', action='store_true', dest="compress")
    parser.add_argument('--no-compress', action='store_false', dest="compress")
    parser.set_defaults(compress=False)
    args = parser.parse_args()

    _compression = "lzf" if args.compress is True else None

    subjects = [os.path.join(args.mpii_root, 'Data', 'Normalized', 'p{:02d}/'.format(_i)) for _i in range(0, 15)]
    hdf_file = h5py.File(os.path.abspath(os.path.join(args.mpii_root, 'mpii_dataset.hdf5')), mode='w')

    for subject_id, subject_path in enumerate(subjects):
        data_files = sorted(glob(os.path.join(subject_path, "*.mat")))
        subject_id = str("s{:03d}".format(subject_id))
        subject_grp = hdf_file.create_group(subject_id)
        data_store_idx = 0
        for mat_fname in tqdm(data_files, desc="Subject {}".format(subject_id)):
            mat = sio.loadmat(mat_fname)
            num_files = mat["filenames"].shape[0]
            for data_idx in range(num_files):
                image_name = "{:0=6d}".format(data_store_idx)
                image_grp = subject_grp.create_group(image_name)
                data_store_idx += 1
                left_image = mat["data"]["left"][0][0]["image"][0][0][data_idx, :]
                left_gaze = mat["data"]["left"][0][0]["gaze"][0][0][data_idx, :]
                left_headpose = mat["data"]["left"][0][0]["pose"][0][0][data_idx, :]
                right_image = mat["data"]["right"][0][0]["image"][0][0][data_idx, :]
                right_gaze = mat["data"]["right"][0][0]["gaze"][0][0][data_idx, :]
                right_headpose = mat["data"]["right"][0][0]["pose"][0][0][data_idx, :]

                left_image = cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR)
                left_image = cv2.resize(left_image, _required_size, cv2.INTER_LANCZOS4)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR)
                right_image = cv2.resize(right_image, _required_size, cv2.INTER_LANCZOS4)

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

                right_rotation_matrix = cv2.Rodrigues(left_headpose)[0]  # ignore the Jackobian matrix
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
