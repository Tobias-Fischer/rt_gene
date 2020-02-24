import glob
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from RTGENEModel_VGG16 import RTGENEModelVGG
from rt_gene.estimate_gaze_base import GazeEstimatorBase
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
from rt_gene.gaze_tools import get_phi_theta_from_euler, limit_yaw

__script_path = os.path.dirname(os.path.realpath(__file__))

_landmark_estimator = LandmarkMethodBase(device_id_facedetection="cuda:0",
                                         checkpoint_path_face=os.path.join(__script_path, "../model_nets/SFD/s3fd_facedetector.pth"),
                                         checkpoint_path_landmark=os.path.join(__script_path, "../model_nets/phase1_wpdc_vdc.pth.tar"),
                                         model_points_file=os.path.join(__script_path, "../model_nets/face_model_68.txt"))
_transform = transforms.Compose([transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])

device_id_gazeestimation = "cuda:0"
_model = RTGENEModelVGG()
_torch_load = torch.load(os.path.join(__script_path, "../model_nets/rt_gene_pytorch_checkpoints/_ckpt_epoch_30.ckpt"))['state_dict']
# for some reason, the state_dict appends _model infront of the states, remove it:
_state_dict = {k[7:]: v for k, v in _torch_load.items()}
_model.load_state_dict(_state_dict)
_model.to(device_id_gazeestimation)
_model.eval()

_img_list = glob.glob(os.path.join(__script_path, "data/", "*.png"))
print(__script_path)
print(_img_list)
print("")


def extract_eye_image_patches(subject):
    le_c, re_c, le_bb, re_bb = subject.get_eye_image_from_landmarks(subject.transformed_eye_landmarks, subject.face_color,
                                                                    _landmark_estimator.eye_image_size)
    subject.left_eye_color = le_c
    subject.right_eye_color = re_c
    subject.left_eye_bb = le_bb
    subject.right_eye_bb = re_bb


try:
    # for file in _img_list:
    _cap = cv2.VideoCapture(2)
    while True:
        # frame = cv2.imread(file)
        # ret = True
        ret, frame = _cap.read()
        if ret:
            cv2.imshow("frame", frame)
            stime = time.time()
            im_width, im_height = frame.shape[1], frame.shape[0]
            _dist_coefficients, _camera_matrix = np.zeros((1, 5)), np.array(
                [[im_height, 0.0, im_width / 2.0], [0.0, im_height, im_height / 2.0], [0.0, 0.0, 1.0]])

            faceboxes = _landmark_estimator.get_face_bb(frame)

            if len(faceboxes) > 0:
                subjects = _landmark_estimator.get_subjects_from_faceboxes(frame, faceboxes)
                subject = subjects[0]
                extract_eye_image_patches(subject)

                if subject.left_eye_color is None or subject.right_eye_color is None:
                    continue

                success, rotation_vector, _ = cv2.solvePnP(_landmark_estimator.model_points,
                                                           subject.landmarks.reshape(len(subject.landmarks), 1, 2),
                                                           cameraMatrix=_camera_matrix,
                                                           distCoeffs=_dist_coefficients,
                                                           flags=cv2.SOLVEPNP_DLS)

                if not success:
                    continue

                roll_pitch_yaw = [-rotation_vector[2], -rotation_vector[0], rotation_vector[1] + np.pi]
                roll_pitch_yaw = limit_yaw(np.array(roll_pitch_yaw).flatten().tolist())

                head_pose = get_phi_theta_from_euler(roll_pitch_yaw)

                _transformed_left = _transform(Image.fromarray(subject.left_eye_color.astype('uint8'), 'RGB')).to(device_id_gazeestimation).unsqueeze(0)
                _transformed_right = _transform(Image.fromarray(subject.right_eye_color.astype('uint8'), 'RGB')).to(device_id_gazeestimation).unsqueeze(0)
                _head_pose = torch.from_numpy(np.array([*head_pose])).to(device_id_gazeestimation).unsqueeze(0).float()

                gaze = _model(_transformed_left, _transformed_right)

                gaze = gaze.detach().cpu().numpy().tolist()
                l_gaze_img = GazeEstimatorBase.visualize_eye_result(subject.left_eye_color, gaze[0])
                r_gaze_img = GazeEstimatorBase.visualize_eye_result(subject.right_eye_color, gaze[0])
                s_gaze_img = np.concatenate((r_gaze_img, l_gaze_img), axis=1)

                cv2.imshow("patches", s_gaze_img)
                cv2.waitKey(1)

except KeyboardInterrupt:
    cv2.destroyAllWindows()
