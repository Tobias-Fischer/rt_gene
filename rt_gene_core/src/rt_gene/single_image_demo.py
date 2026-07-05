import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from rt_gene.estimate_gaze_pytorch import GazeEstimator
from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
import rt_gene.gaze_tools as gaze_tools
from rt_gene.gaze_tools_standalone import euler_from_matrix
from rt_gene.tracker_generic import TrackedSubject
from rt_gene_core import transforms as frame_transforms
from rt_gene_core.paths import model_path


DEFAULT_GAZE_MODEL = "gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model"


def default_camera_matrix(width, height, focal_length_px=None):
    focal = float(focal_length_px or max(width, height))
    return np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])


def resolve_model_files(files):
    resolved = []
    for item in files:
        path = Path(item)
        resolved.append(str(path if path.is_absolute() else model_path(item, writable=True)))
    return resolved


def estimate_head_pose(method, subject, camera_matrix):
    success, rotation, translation, _ = cv2.solvePnPRansac(
        method.model_points,
        subject.landmarks.reshape(len(method.model_points), 1, 2),
        cameraMatrix=camera_matrix,
        distCoeffs=np.zeros(5),
        flags=cv2.SOLVEPNP_DLS,
    )
    if not success:
        raise RuntimeError("Could not estimate head pose from detected landmarks")

    rotation[0] += method.head_pitch
    rotation_matrix, _ = cv2.Rodrigues(rotation)
    rotation_matrix = np.matmul(rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation_matrix
    return np.array(euler_from_matrix(matrix)), (translation / 1000.0).reshape(3), matrix


def run(image_path, device="auto", focal_length_px=None, model_files=None):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    method = LandmarkMethodBase(device)
    faceboxes = method.get_face_bb(image)
    if not faceboxes:
        raise RuntimeError("No face detected")

    camera_matrix = default_camera_matrix(image.shape[1], image.shape[0], focal_length_px)
    subjects = method.get_subjects_from_faceboxes(image, faceboxes)
    estimator = GazeEstimator(device, resolve_model_files(model_files or [DEFAULT_GAZE_MODEL]))

    results = []
    for index, subject in enumerate(subjects):
        subject.left_eye_color, subject.right_eye_color, _, _ = TrackedSubject.get_eye_image_from_landmarks(
            subject, method.eye_image_size
        )
        if subject.left_eye_color is None or subject.right_eye_color is None:
            continue

        head_rpy, translation, head_matrix = estimate_head_pose(method, subject, camera_matrix)
        euler = list(euler_from_matrix(np.dot(np.array(frame_transforms.camera_to_ros), head_matrix)))
        phi_head, theta_head = gaze_tools.get_phi_theta_from_euler(gaze_tools.limit_yaw(euler))
        gaze = estimator.estimate_gaze_twoeyes(
            [estimator.input_from_image(subject.left_eye_color)],
            [estimator.input_from_image(subject.right_eye_color)],
            [[theta_head, phi_head]],
        )[0]
        results.append({
            "subject_id": index,
            "face_box": [float(v) for v in subject.box],
            "head_pose_rpy_rad": [float(v) for v in head_rpy],
            "head_translation_m": [float(v) for v in translation],
            "gaze_theta_phi_rad": [float(v) for v in gaze],
        })

    if not results:
        raise RuntimeError("Face detected, but eye crops were invalid")
    return {
        "image": str(image_path),
        "device": str(method.device),
        "subjects": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run RT-GENE on one image and print head pose/gaze JSON.")
    parser.add_argument("image")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--focal-length-px", type=float)
    parser.add_argument("--model", action="append", dest="models", help="Gaze model file/name. Repeat to ensemble.")
    args = parser.parse_args(argv)

    try:
        print(json.dumps(run(args.image, args.device, args.focal_length_px, args.models), indent=2))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
