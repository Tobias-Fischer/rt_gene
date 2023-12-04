import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


class RTGENEH5Dataset(data.Dataset):

    def __init__(self, h5_file, subject_list=None, transform=None):
        self._h5_file = h5_file
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((36, 60), Image.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        _wanted_subjects = ["s{:03d}".format(_i) for _i in subject_list]

        for grp_s_n in tqdm(_wanted_subjects, desc="Loading subject metadata..."):  # subjects
            for grp_i_n, grp_i in h5_file[grp_s_n].items():  # images
                if "left" in grp_i.keys() and "right" in grp_i.keys() and "label" in grp_i.keys():
                    left_dataset = grp_i["left"]
                    right_datset = grp_i['right']

                    assert len(left_dataset) == len(right_datset), "Dataset left/right images aren't equal length"
                    for _i in range(len(left_dataset)):
                        self._subject_labels.append(["/" + grp_s_n + "/" + grp_i_n, _i])

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        _sample = self._subject_labels[index]
        assert type(_sample[0]) == str, "Sample not found at index {}".format(index)
        _left_img = self._h5_file[_sample[0] + "/left"][_sample[1]][()]
        _right_img = self._h5_file[_sample[0] + "/right"][_sample[1]][()]
        label_data = self._h5_file[_sample[0]+"/label"][()]
        _groud_truth_headpose = label_data[0][()].astype(float)
        _ground_truth_gaze = label_data[1][()].astype(float)

        # Load data and get label
        _transformed_left = self._transform(Image.fromarray(_left_img, 'RGB'))
        _transformed_right = self._transform(Image.fromarray(_right_img, 'RGB'))

        return _transformed_left, _transformed_right, _groud_truth_headpose, _ground_truth_gaze


class RTGENEFileDataset(data.Dataset):

    def __init__(self, root_path, subject_list=None, transform=None):
        self._root_path = root_path
        self._transform = transform
        self._subject_labels = []

        assert subject_list is not None, "Must pass a list of subjects to load the data for"

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        subject_path = [os.path.join(root_path, "s{:03d}_glasses/".format(_i)) for _i in subject_list]

        for subject_data in subject_path:
            with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
                _lines = f.readlines()
                for line in _lines:
                    split = line.split(",")
                    left_img_path = os.path.join(subject_data, "inpainted/left_new/", "left_{:0=6d}_rgb.png".format(int(split[0])))
                    right_img_path = os.path.join(subject_data, "inpainted/right_new/", "right_{:0=6d}_rgb.png".format(int(split[0])))
                    if os.path.exists(left_img_path) and os.path.exists(right_img_path):
                        head_phi = float(split[1].strip()[1:])
                        head_theta = float(split[2].strip()[:-1])
                        gaze_phi = float(split[3].strip()[1:])
                        gaze_theta = float(split[4].strip()[:-1])
                        self._subject_labels.append([left_img_path, right_img_path, head_phi, head_theta, gaze_phi, gaze_theta])

        print("=> Loaded metadata for {} images".format(len(self._subject_labels)))

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        _sample = self._subject_labels[index]
        _groud_truth_headpose = [_sample[2], _sample[3]]
        _ground_truth_gaze = [_sample[4], _sample[5]]

        # Load data and get label
        _left_img = np.array(Image.open(os.path.join(self._root_path, _sample[0])).convert('RGB'))
        _right_img = np.array(Image.open(os.path.join(self._root_path, _sample[1])).convert('RGB'))

        _transformed_left = self._transform(Image.fromarray(_left_img, 'RGB'))
        _transformed_right = self._transform(Image.fromarray(_right_img, 'RGB'))

        return _transformed_left, _transformed_right, np.array(_groud_truth_headpose, dtype=float), np.array(_ground_truth_gaze, dtype=float)
