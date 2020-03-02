import os

import h5py
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
from tqdm import tqdm


class RTGENEDataset(data.Dataset):

    def __init__(self, h5_file, subject_list=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), transform=None):
        self._h5_file = h5_file
        self._transform = transform
        self._subject_labels = []

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC),
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
                        self._subject_labels.append([grp_i.name, _i])

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        _sample = self._subject_labels[index]
        _left_img = self._h5_file[_sample[0] + "/left"][_sample[1]][()]
        _right_img = self._h5_file[_sample[0] + "/right"][_sample[1]][()]
        label_data = self._h5_file[_sample[0]+"/label"][()]
        _groud_truth_headpose = label_data[0][()].astype(np.float32)
        _ground_truth_gaze = label_data[1][()].astype(np.float32)

        # Load data and get label
        _transformed_left = self._transform(Image.fromarray(_left_img, 'RGB'))
        _transformed_right = self._transform(Image.fromarray(_right_img, 'RGB'))

        return _transformed_left, _transformed_right, _groud_truth_headpose, _ground_truth_gaze


if __name__ == "__main__":
    from tqdm import trange

    h5file = os.path.abspath("/home/ahmed/Documents/RT_GENE/dataset.hdf5")
    _ds = RTGENEDataset(h5_file=h5py.File(h5file, 'r'), subject_list=[0])

    for i in trange(1000):
        left, right, head_pose, gaze_pose = _ds[i]
