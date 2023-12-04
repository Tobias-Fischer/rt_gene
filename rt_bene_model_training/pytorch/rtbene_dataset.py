import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm
import h5py


class RTBENEH5Dataset(data.Dataset):

    def __init__(self, h5_pth, subject_list=None, transform=None, loader_desc="train"):
        self._h5_file = h5_pth
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((36, 60), transforms.InterpolationMode.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])

        _wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        with h5py.File(self._h5_file, mode="r") as h5_file:
            for grp_s_n in tqdm(_wanted_subjects, desc="Loading ({}) subject metadata...".format(loader_desc), position=0):  # subjects
                for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                    if "left" in grp_i.keys() and "right" in grp_i.keys() and "label" in grp_i.keys():
                        left_dataset = grp_i["left"]
                        right_datset = grp_i['right']

                        assert len(left_dataset) == len(
                            right_datset), "Weird: Dataset left/right images aren't equal length"
                        for _i in range(len(left_dataset)):
                            self._subject_labels.append(["/" + grp_s_n + "/" + grp_i_n, _i])

    @staticmethod
    def get_class_weights(h5_file, subject_list):
        positive = 0
        total = 0
        _wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        for grp_s_n in tqdm(_wanted_subjects, desc="Loading class weights...", position=0):
            for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                if "left" in grp_i.keys() and "right" in grp_i.keys() and "label" in grp_i.keys():
                    label = grp_i["label"][()][0]
                    if label == 1.0:
                        positive = positive + 1
                    total = total + 1

        negative = total - positive
        weight_for_0 = (negative + positive) / negative
        weight_for_1 = (negative + positive) / positive
        return {0: weight_for_0, 1: weight_for_1}

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        sample = self._subject_labels[index]

        with h5py.File(self._h5_file, mode="r") as h5_file:
            left_img = h5_file[sample[0] + "/left"][sample[1]][()][0]
            right_img = h5_file[sample[0] + "/right"][sample[1]][()][0]
            label = h5_file[sample[0] + "/label"][()].astype(float)

            # Load data and get label
            transformed_left_img = self._transform(Image.fromarray(left_img, 'RGB'))
            transformed_right_img = self._transform(Image.fromarray(right_img, 'RGB'))

            return transformed_left_img, transformed_right_img, label
