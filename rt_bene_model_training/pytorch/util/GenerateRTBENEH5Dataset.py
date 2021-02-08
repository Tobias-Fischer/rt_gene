from __future__ import print_function, division, absolute_import

import argparse
import os

import h5py
import numpy as np
from PIL import Image, ImageFilter, ImageOps
from torchvision import transforms
from tqdm import tqdm

script_path = os.path.dirname(os.path.realpath(__file__))


_required_size = (224, 224)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Estimate gaze from images')
    parser.add_argument('--rt_bene_root', type=str, required=True, nargs='?',
                        help='Path to the base directory of RT_BENE')
    parser.add_argument('--compress', action='store_true', dest="compress", help="Whether to use LZF compression or not")
    parser.add_argument('--no-compress', action='store_false', dest="compress")
    parser.set_defaults(compress=False)
    args = parser.parse_args()

    _compression = "lzf" if args.compress is True else None

    subject_path = [os.path.join(args.rt_bene_root, "s{:03d}_noglasses/".format(_i)) for _i in range(0, 17)]

    with h5py.File(os.path.abspath(os.path.join(args.rt_bene_root, "rtbene_dataset.hdf5")), mode='w') as hdf_file:
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
                            image_data = np.array([Image.open(image_path).resize(_required_size)])
                            image_grp.create_dataset("image", data=image_data, compression=_compression)
                            image_grp.create_dataset("label", data=[label])
