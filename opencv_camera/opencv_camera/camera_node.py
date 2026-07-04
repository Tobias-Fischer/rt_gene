from pathlib import Path
import copy

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from opencv_camera.calibration import load_camera_info


class OpenCVCameraNode(Node):
    def __init__(self):
        super().__init__("opencv_camera")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("video_file", "")
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("camera_name", "camera")
        self.declare_parameter("frame_id", "camera_optical_frame")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("loop", False)

        video_file = self.get_parameter("video_file").value
        source = str(Path(video_file).expanduser()) if video_file else int(self.get_parameter("camera_index").value)
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")

        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)
        if width > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height > 0:
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps > 0:
            self.capture.set(cv2.CAP_PROP_FPS, fps)

        self.frame_id = self.get_parameter("frame_id").value.lstrip("/") or "camera_optical_frame"
        actual_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
        actual_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
        self.camera_info = load_camera_info(
            self.get_parameter("calibration_file").value,
            actual_width,
            actual_height,
            self.get_parameter("camera_name").value,
            self.frame_id,
        )
        self.loop = bool(self.get_parameter("loop").value)
        self.bridge = CvBridge()
        self.image_pub = self.create_publisher(Image, "image_raw", qos_profile_sensor_data)
        self.info_pub = self.create_publisher(CameraInfo, "camera_info", qos_profile_sensor_data)
        self.timer = self.create_timer(1.0 / max(fps, 1.0), self.publish_frame)

    def publish_frame(self):
        ok, frame = self.capture.read()
        if not ok and self.loop:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        if not ok:
            self.get_logger().warn("No frame available", throttle_duration_sec=5.0)
            return

        header = self.camera_info.header
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.frame_id
        image_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        image_msg.header = header
        info_msg = copy.deepcopy(self.camera_info)
        info_msg.header = header
        self.image_pub.publish(image_msg)
        self.info_pub.publish(info_msg)

    def destroy_node(self):
        if hasattr(self, "capture"):
            self.capture.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = OpenCVCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
