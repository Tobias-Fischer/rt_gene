import os

import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms


class RTGENEDataset(data.Dataset):

    def __init__(self, root_path, subject_list=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), transform=None):
        self._root_path = root_path
        self._transform = transform
        self._subject_labels = []

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

        return _transformed_left, _transformed_right, np.array(_groud_truth_headpose, dtype=np.float32), np.array(_ground_truth_gaze, dtype=np.float32)


if __name__ == "__main__":
    from tqdm import trange

    __script_path = os.path.dirname(os.path.realpath(__file__))
    _root_path = os.path.abspath("/home/ahmed/Documents/RT_GENE/")
    print(_root_path)
    _ds = RTGENEDataset(root_path=_root_path, subject_list=[0])

    for i in trange(len(_ds)):
        left, right, head_pose, gaze_pose = _ds[i]
