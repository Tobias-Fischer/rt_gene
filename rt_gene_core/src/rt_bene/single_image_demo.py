import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

from rt_gene_core.paths import demo_image_path, model_path


DEFAULT_BLINK_MODEL = "blink_model_pytorch_vgg16_allsubjects1.model"


def resolve_model_files(files):
    resolved = []
    for item in files:
        path = Path(item)
        resolved.append(str(path if path.is_absolute() else model_path(item, writable=True)))
    return resolved


def _read_image(path):
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _direct_eye_pair(left_eye_path, right_eye_path):
    return [{
        "subject_id": 0,
        "left_eye": _read_image(left_eye_path),
        "right_eye": _read_image(right_eye_path),
        "source": {
            "left_eye": str(Path(left_eye_path)),
            "right_eye": str(Path(right_eye_path)),
        },
    }]


def _eye_pairs_from_face(image_path, landmark_device):
    image_path = Path(image_path)
    image = _read_image(image_path)

    from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
    from rt_gene.tracker_generic import TrackedSubject

    method = LandmarkMethodBase(landmark_device)
    faceboxes = method.get_face_bb(image)
    if not faceboxes:
        raise RuntimeError("No face detected")

    pairs = []
    for index, subject in enumerate(method.get_subjects_from_faceboxes(image, faceboxes)):
        left, right, _, _ = TrackedSubject.get_eye_image_from_landmarks(subject, method.eye_image_size)
        if left is None or right is None:
            continue
        pairs.append({
            "subject_id": index,
            "left_eye": left,
            "right_eye": right,
            "face_box": [float(value) for value in subject.box],
            "source": {"image": str(image_path)},
        })

    if not pairs:
        raise RuntimeError("Face detected, but eye crops were invalid")
    return pairs, str(method.device)


def run(
    image_path=None,
    left_eye_path=None,
    right_eye_path=None,
    device="auto",
    landmark_device="cpu",
    model_files=None,
    model_type="vgg16",
    threshold=0.425,
):
    if bool(left_eye_path) != bool(right_eye_path):
        raise ValueError("--left-eye and --right-eye must be provided together")

    if left_eye_path and right_eye_path:
        pairs = _direct_eye_pair(left_eye_path, right_eye_path)
        resolved_landmark_device = None
    else:
        pairs, resolved_landmark_device = _eye_pairs_from_face(image_path or demo_image_path(), landmark_device)

    from rt_bene.estimate_blink_pytorch import BlinkEstimatorPytorch

    estimator = BlinkEstimatorPytorch(
        device_id_blink=device,
        model_files=resolve_model_files(model_files or [DEFAULT_BLINK_MODEL]),
        model_type=model_type,
        threshold=threshold,
    )
    left_inputs, right_inputs = [], []
    for pair in pairs:
        left, right = estimator.inputs_from_images(pair["left_eye"], pair["right_eye"])
        left_inputs.append(left)
        right_inputs.append(right)

    probabilities = np.asarray(estimator.predict(left_inputs, right_inputs)).reshape(-1)
    subjects = []
    for pair, probability in zip(pairs, probabilities):
        subject = {
            "subject_id": pair["subject_id"],
            "blink_probability": float(probability),
            "blink": bool(probability >= estimator.threshold),
            "source": pair["source"],
        }
        if "face_box" in pair:
            subject["face_box"] = pair["face_box"]
        subjects.append(subject)

    return {
        "device": str(estimator.device_id),
        "landmark_device": resolved_landmark_device,
        "threshold": float(estimator.threshold),
        "subjects": subjects,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run RT-BENE blink estimation and print JSON.")
    parser.add_argument("image", nargs="?", default=str(demo_image_path()), help="Face image used when eye patches are not provided.")
    parser.add_argument("--left-eye", help="Left eye crop. Must be used with --right-eye.")
    parser.add_argument("--right-eye", help="Right eye crop. Must be used with --left-eye.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--landmark-device", default="cpu")
    parser.add_argument("--model", action="append", dest="models", help="Blink model file/name. Repeat to ensemble.")
    parser.add_argument("--model-type", default="vgg16")
    parser.add_argument("--threshold", type=float, default=0.425)
    args = parser.parse_args(argv)

    try:
        print(json.dumps(
            run(
                args.image,
                args.left_eye,
                args.right_eye,
                args.device,
                args.landmark_device,
                args.models,
                args.model_type,
                args.threshold,
            ),
            indent=2,
        ))
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
