from pathlib import Path

from sensor_msgs.msg import CameraInfo
import yaml


def default_camera_info(width, height, camera_name, frame_id):
    msg = CameraInfo()
    msg.width = int(width)
    msg.height = int(height)
    msg.header.frame_id = frame_id
    msg.distortion_model = "plumb_bob"
    fx = fy = float(height or width or 1)
    cx = float(width) / 2.0
    cy = float(height) / 2.0
    msg.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
    msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
    return msg


def load_camera_info(path, width, height, camera_name, frame_id):
    if not path:
        return default_camera_info(width, height, camera_name, frame_id)

    with Path(path).expanduser().open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)

    msg = CameraInfo()
    msg.width = int(data.get("image_width", width))
    msg.height = int(data.get("image_height", height))
    msg.header.frame_id = frame_id
    msg.distortion_model = data.get("distortion_model", "plumb_bob")
    msg.d = [float(v) for v in data["distortion_coefficients"]["data"]]
    msg.k = [float(v) for v in data["camera_matrix"]["data"]]
    msg.r = [float(v) for v in data["rectification_matrix"]["data"]]
    msg.p = [float(v) for v in data["projection_matrix"]["data"]]
    return msg
