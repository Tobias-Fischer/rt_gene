import os

import cv2
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase


class RTGENEDataset(data.Dataset):
    __script_path = os.path.dirname(os.path.realpath(__file__))
    __landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                              checkpoint_path_face=os.path.join(__script_path, "../model_nets/SFD/s3fd_facedetector.pth"),
                                              checkpoint_path_landmark=os.path.join(__script_path, "../model_nets/phase1_wpdc_vdc.pth.tar"),
                                              model_points_file=os.path.join(__script_path, "../model_nets/face_model_68.txt"))

    def __init__(self, root_path, subject_list=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), transform=None):
        self._root_path = root_path
        self._transform = transform
        self._subject_labels = []

        if self._transform is None:
            self._transform = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        subject_path = [os.path.join(root_path, "s{:03d}_glasses/".format(i)) for i in subject_list]

        for subject_data in subject_path:
            with open(os.path.join(subject_data, "label_combined.txt"), "r") as f:
                _lines = f.readlines()
                for l in _lines:
                    split = l.split(",")
                    img_name = os.path.join(subject_data, "inpainted/face_after_inpainting/", "{:0=6d}.png".format(int(split[0])))
                    gaze_phi = float(split[3].strip()[1:])
                    gaze_theta = float(split[4].strip()[:-1])
                    self._subject_labels.append([img_name, gaze_phi, gaze_theta])

        print("=> Loaded metadata for {} images".format(len(self._subject_labels)))

    @staticmethod
    def _extract_eye_image_patches(subject):
        le_c, re_c, le_bb, re_bb = subject.get_eye_image_from_landmarks(subject.transformed_landmarks, subject.face_color,
                                                                        RTGENEDataset.__landmark_estimator.eye_image_size)
        subject.left_eye_color = le_c
        subject.right_eye_color = re_c
        subject.left_eye_bb = le_bb
        subject.right_eye_bb = re_bb

    def _visualise_eye_patches(self, subject, pose, index):
        if not hasattr(self, "_test_transform"):
            self._test_transform = transforms.Compose([transforms.Resize((224, 224), Image.BICUBIC)])

        _transformed_left = self._test_transform(Image.fromarray(subject.left_eye_color.astype('uint8'), 'RGB'))
        _transformed_right = self._test_transform(Image.fromarray(subject.right_eye_color.astype('uint8'), 'RGB'))
        s_gaze_img = np.concatenate((_transformed_right, _transformed_left), axis=1)

        s_gaze_img = cv2.putText(s_gaze_img, 'Theta: {:+.2f}, Phi: {:+.2f}, Index: {:07d}'.format(pose[0], pose[1], index), (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                 0.6, (255, 255, 255), 1)
        cv2.imshow("face", subject.face_color)
        cv2.imshow("patches", s_gaze_img)
        cv2.waitKey(1)

    def __len__(self):
        return len(self._subject_labels)

    def __getitem__(self, index):
        try:
            # Select sample
            _sample = self._subject_labels[index]
            _ground_truth_gaze = [_sample[1], _sample[2]]

            # Load data and get label
            _img = np.array(Image.open(os.path.join(self._root_path, _sample[0])).convert('RGB'))
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            im_width, im_height = _img.shape[1], _img.shape[0]
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

            faceboxes = RTGENEDataset.__landmark_estimator.get_face_bb(_img)

            if len(faceboxes) > 0:
                subjects = RTGENEDataset.__landmark_estimator.get_subjects_from_faceboxes(_img, faceboxes)
                subject = subjects[0]
                self._extract_eye_image_patches(subject)

                if subject.left_eye_color is None or subject.right_eye_color is None:
                    raise ValueError("Unable to find eye patches for img: {}".format(os.path.join(self._root_path, _sample[0])))

                success, rotation_vector, _ = cv2.solvePnP(RTGENEDataset.__landmark_estimator.model_points,
                                                           subject.marks.reshape(len(subject.marks), 1, 2),
                                                           cameraMatrix=_camera_matrix,
                                                           distCoeffs=_dist_coefficients,
                                                           flags=cv2.SOLVEPNP_DLS)

                if not success:
                    raise ValueError("SolvPnP Not successful for for img: {}".format(os.path.join(self._root_path, _sample[0])))

                # TODO: change to whatever the latest in the standalone is
                roll_pitch_yaw = np.array([-rotation_vector[2], -rotation_vector[0], rotation_vector[1] + np.pi]).flatten()
                # roll_pitch_yaw = limit_yaw(np.array(roll_pitch_yaw).flatten().tolist())
                # head_pose = get_phi_theta_psi_from_euler(roll_pitch_yaw)

                _transformed_left = self._transform(Image.fromarray(subject.left_eye_color.astype('uint8'), 'RGB'))
                _transformed_right = self._transform(Image.fromarray(subject.right_eye_color.astype('uint8'), 'RGB'))
                # self._visualise_eye_patches(subject, _ground_truth_gaze, index)

                return _transformed_left, _transformed_right, np.array(_ground_truth_gaze, dtype=np.float32), roll_pitch_yaw
            else:
                raise ValueError("No Face found for img: {}".format(os.path.join(self._root_path, _sample[0])))
        except ValueError:
            return None


if __name__ == "__main__":
    import nonechucks as nc
    from tqdm import trange

    _ds = RTGENEDataset(root_path="../../data", subject_list=list(range(0, 17)))
    _data_loader = nc.SafeDataLoader(nc.SafeDataset(_ds), batch_size=1, shuffle=False)

    _data_loader_iter = iter(_data_loader)
    for i in trange(100):
        batch = next(_data_loader_iter)
