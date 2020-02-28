from __future__ import print_function, division, absolute_import

import argparse
import os

import h5py
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from tqdm import tqdm

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase

script_path = os.path.dirname(os.path.realpath(__file__))

# Tobias randomly crops and resizes the image 10 times in the `prepare_dataset.m` along side two blurring stages, grayscaling and histogram normalisation
_required_size = (224, 224)
_transforms_list = [transforms.RandomResizedCrop(size=_required_size, scale=(0.85, 1.0)),  # equivilant to random 5px from each edge
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
                    lambda x: x.filter(ImageFilter.GaussianBlur(radius=1)).resize(_required_size),
                    lambda x: x.filter(ImageFilter.GaussianBlur(radius=3)).resize(_required_size),
                    lambda x: ImageOps.equalize(x).resize(_required_size)]  # histogram equalisation


def load_and_augment(file_path):
    image = Image.open(file_path).resize(_required_size)
    augmented_images = [np.array(trans(image)) for trans in _transforms_list]
    augmented_images.append(np.array(image))

    return np.array(augmented_images)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('--rt_gene_root', type=str, required=True,
                        nargs='?', help='Path to the base directory of RT_GENE')

    landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                            checkpoint_path_face=os.path.abspath(os.path.join(script_path, "../../rt_gene/model_nets/SFD/s3fd_facedetector.pth")),
                                            checkpoint_path_landmark=os.path.abspath(
                                                os.path.join(script_path, "../../rt_gene/model_nets/phase1_wpdc_vdc.pth.tar")),
                                            model_points_file=os.path.abspath(os.path.join(script_path, "../../rt_gene/model_nets/face_model_68.txt")))

    args = parser.parse_args()

    subject_path = [os.path.join(args.rt_gene_root, "s{:03d}_glasses/".format(_i)) for _i in list(range(0, 17))]

    hdf_file = h5py.File(os.path.abspath(os.path.join(args.rt_gene_root, 'dataset.hdf5')), mode='w')
    for subject_id, subject_data in enumerate(subject_path):
        subject_id = str("s{:03d}".format(subject_id))
        subject_grp = hdf_file.create_group(subject_id)
        with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
            _lines = f.readlines()

            for line in tqdm(_lines, desc="Subject {}".format(subject_id)):

                split = line.split(",")
                image_name = "{:0=6d}".format(int(split[0]))
                image_grp = subject_grp.create_group(image_name)
                left_img_path = os.path.join(subject_data, "inpainted/left_new/", "left_{:0=6d}_rgb.png".format(int(split[0])))
                right_img_path = os.path.join(subject_data, "inpainted/right_new/", "right_{:0=6d}_rgb.png".format(int(split[0])))
                if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                    head_phi = float(split[1].strip()[1:])
                    head_theta = float(split[2].strip()[:-1])
                    gaze_phi = float(split[3].strip()[1:])
                    gaze_theta = float(split[4].strip()[:-1])
                    labels = [(head_phi, head_theta), (gaze_phi, gaze_theta)]
                    left_image_grp = image_grp.create_group("left")
                    right_image_grp = image_grp.create_group("right")

                    left_data = load_and_augment(left_img_path)
                    right_data = load_and_augment(right_img_path)
                    left_image_grp.create_dataset("data", data=left_data, compression="gzip")
                    right_image_grp.create_dataset("data", data=right_data, compression="gzip")
                    image_grp.create_dataset("label", data=labels)

    hdf_file.flush()
    hdf_file.close()
