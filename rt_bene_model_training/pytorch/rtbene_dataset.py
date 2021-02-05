import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


class RTBENEH5Dataset(data.Dataset):

    def __init__(self, h5_file, subject_list=None, transform=None, loader_desc="train"):
        self._h5_file = h5_file
        self._transform = transform
        self._subject_labels = []
        self._positive_labels = 0
        self._total_labels = 0

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225])])

        _wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        for grp_s_n in tqdm(_wanted_subjects, desc="Loading ({}) subject metadata...".format(loader_desc)):  # subjects
            for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                if "image" in grp_i.keys() and "label" in grp_i.keys():
                    image_dataset = grp_i["image"]
                    label = grp_i["label"]
                    if label == 1.0:
                        self._positive_labels = self._positive_labels + 1
                    self._total_labels = self._total_labels + 1

                    for _i in range(len(image_dataset)):
                        self._subject_labels.append(["/" + grp_s_n + "/" + grp_i_n, _i])

    @staticmethod
    def get_class_weights(h5_file, subject_list):
        positive = 0
        total = 0
        _wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        for grp_s_n in tqdm(_wanted_subjects, desc="Loading class weights..."):
            for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                if "image" in grp_i.keys() and "label" in grp_i.keys():
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
        _sample = self._subject_labels[index]
        assert type(_sample[0]) == str, "Sample not found at index {}".format(index)
        _img = self._h5_file[_sample[0] + "/image"][_sample[1]][()]
        label_data = self._h5_file[_sample[0] + "/label"][()].astype(np.float32)

        # Load data and get label
        _transformed_img = self._transform(Image.fromarray(_img, 'RGB'))

        return _transformed_img, label_data
