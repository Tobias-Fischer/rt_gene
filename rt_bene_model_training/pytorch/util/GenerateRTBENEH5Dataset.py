from __future__ import print_function, division, absolute_import

import argparse
import os

import h5py
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from tqdm import tqdm

script_path = os.path.dirname(os.path.realpath(__file__))

# Augmentations following `prepare_dataset.m`: randomly crop and resize the image 10 times,
# along side two blurring stages, grayscaling and histogram normalisation
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


def load_and_augment(file_path, augment=False):
    image = Image.open(file_path).resize(_required_size)
    augmented_images = [np.array(trans(image)) for trans in _transforms_list if augment is True]
    augmented_images.append(np.array(image))

    return np.array(augmented_images, dtype=np.uint8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('--rt_bene_root', type=str, required=True, nargs='?', help='Path to the base directory of RT_GENE')
    parser.add_argument('--augment_dataset', type=bool, required=False, default=False, help="Whether to augment the dataset with predefined transforms")
    parser.add_argument('--compress', action='store_true', dest="compress")
    parser.add_argument('--no-compress', action='store_false', dest="compress")
    parser.set_defaults(compress=False)
    args = parser.parse_args()

    _compression = "lzf" if args.compress is True else None

    subject_path = [os.path.join(args.rt_bene_root, "s{:03d}_noglasses/".format(_i)) for _i in range(0, 17)]

    hdf_file = h5py.File(os.path.abspath(os.path.join(args.rt_bene_root, "rtbene_dataset.hdf5")), mode='w')
    for subject_id, subject_data in enumerate(subject_path):
        subject_id = str("s{:03d}".format(subject_id))
        subject_grp = hdf_file.create_group(subject_id)
        with open(os.path.join(args.rt_bene_root, "{}_blink_labels.csv".format(subject_id)), "r") as f:
            _lines = f.readlines()

            for line in tqdm(_lines, desc="Subject {}".format(subject_id)):

                split = line.split(",")
                image_name = split[0]
                image_grp = subject_grp.create_group(image_name)
                image_path = os.path.join(subject_data, "natural/left/", "{}".format(split[0]))
                if os.path.exists(image_path):
                    label = float(split[1].strip("\n"))
                    if label != 0.5:  # paper removed 0.5s
                        image_data = load_and_augment(image_path, augment=args.augment_dataset)
                        image_grp.create_dataset("image", data=image_data, compression=_compression)
                        image_grp.create_dataset("label", data=[label])

    hdf_file.flush()
    hdf_file.close()
