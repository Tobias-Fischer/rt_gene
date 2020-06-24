# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.backends import cudnn as cudnn
from tqdm import tqdm

from rt_gene.download_tools import download_external_landmark_models

# noinspection PyUnresolvedReferences
from rt_gene import gaze_tools as gaze_tools
from rt_gene.SFD.sfd_detector import SFDDetector
from rt_gene.ThreeDDFA.ddfa import ToTensorGjz, NormalizeGjz
from rt_gene.ThreeDDFA.inference import crop_img, predict_68pts, parse_roi_box_from_bbox, parse_roi_box_from_landmark
from rt_gene.tracker_generic import TrackedSubject

facial_landmark_transform = transforms.Compose([ToTensorGjz(), NormalizeGjz(mean=127.5, std=128)])


class LandmarkMethodBase(object):
    def __init__(self, device_id_facedetection, checkpoint_path_face=None, checkpoint_path_landmark=None, model_points_file=None):
        download_external_landmark_models()
        self.model_size_rescale = 16.0
        self.head_pitch = 0.0
        self.interpupillary_distance = 0.058
        self.eye_image_size = (60, 36)

        tqdm.write("Using device {} for face detection.".format(device_id_facedetection))

        self.device = device_id_facedetection
        self.face_net = SFDDetector(device=device_id_facedetection, path_to_detector=checkpoint_path_face)
        self.facial_landmark_nn = self.load_face_landmark_model(checkpoint_path_landmark)

        self.model_points = self.get_full_model_points(model_points_file)

    def load_face_landmark_model(self, checkpoint_fp=None):
        import rt_gene.ThreeDDFA.mobilenet_v1 as mobilenet_v1
        if checkpoint_fp is None:
            import rospkg
            checkpoint_fp = rospkg.RosPack().get_path('rt_gene') + '/model_nets/phase1_wpdc_vdc.pth.tar'
        arch = 'mobilenet_1'

        checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
        model = getattr(mobilenet_v1, arch)(num_classes=62)  # 62 = 12(pose) + 40(shape) +10(expression)

        model_dict = model.state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        cudnn.benchmark = True
        model = model.to(self.device)
        model.eval()
        return model

    def get_full_model_points(self, model_points_file=None):
        """Get all 68 3D model points from file"""
        raw_value = []
        if model_points_file is None:
            import rospkg
            model_points_file = rospkg.RosPack().get_path('rt_gene') + '/model_nets/face_model_68.txt'

        with open(model_points_file) as f:
            for line in f:
                raw_value.append(line)
        model_points = np.array(raw_value, dtype=np.float32)
        model_points = np.reshape(model_points, (3, -1)).T

        # index the expansion of the model based.
        model_points = model_points * (self.interpupillary_distance * self.model_size_rescale)

        return model_points

    def get_face_bb(self, image):
        faceboxes = []
        fraction = 4.0
        image = cv2.resize(image, (0, 0), fx=1.0 / fraction, fy=1.0 / fraction)
        detections = self.face_net.detect_from_image(image)

        for result in detections:
            # scale back up to image size
            box = result[:4]
            confidence = result[4]

            if gaze_tools.box_in_image(box, image) and confidence > 0.6:
                box = [x * fraction for x in box]  # scale back up
                diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
                offset_y = int(abs(diff_height_width / 2))
                box_moved = gaze_tools.move_box(box, [0, offset_y])

                # Make box square.
                facebox = gaze_tools.get_square_box(box_moved)
                faceboxes.append(facebox)

        return faceboxes

    @staticmethod
    def visualize_headpose_result(face_image, est_headpose):
        """Here, we take the original eye eye_image and overlay the estimated headpose."""
        output_image = np.copy(face_image)

        center_x = output_image.shape[1] / 2
        center_y = output_image.shape[0] / 2

        endpoint_x, endpoint_y = gaze_tools.get_endpoint(est_headpose[1], est_headpose[0], center_x, center_y, 100)

        cv2.line(output_image, (int(center_x), int(center_y)), (int(endpoint_x), int(endpoint_y)), (0, 0, 255), 3)
        return output_image

    def ddfa_forward_pass(self, color_img, roi_box_list):
        img_step = [crop_img(color_img, roi_box) for roi_box in roi_box_list]
        img_step = [cv2.resize(img, dsize=(120, 120), interpolation=cv2.INTER_LINEAR) for img in img_step]
        _input = torch.cat([facial_landmark_transform(img).unsqueeze(0) for img in img_step], 0)
        with torch.no_grad():
            _input = _input.to(self.device)
            param = self.facial_landmark_nn(_input).cpu().numpy().astype(np.float32)

        return [predict_68pts(p.flatten(), roi_box) for p, roi_box in zip(param, roi_box_list)]

    def get_subjects_from_faceboxes(self, color_img, faceboxes):
        face_images = [gaze_tools.crop_face_from_image(color_img, b) for b in faceboxes]
        subjects = []
        roi_box_list = [parse_roi_box_from_bbox(facebox) for facebox in faceboxes]
        initial_pts68_list = self.ddfa_forward_pass(color_img, roi_box_list)
        roi_box_refined_list = [parse_roi_box_from_landmark(initial_pts68) for initial_pts68 in initial_pts68_list]
        pts68_list = self.ddfa_forward_pass(color_img, roi_box_refined_list)

        for pts68, face_image, facebox in zip(pts68_list, face_images, faceboxes):
            np_landmarks = np.array((pts68[0], pts68[1])).T
            subjects.append(TrackedSubject(np.array(facebox), face_image, np_landmarks))
        return subjects
