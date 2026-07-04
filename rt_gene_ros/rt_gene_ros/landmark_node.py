import cv2
from cv_bridge import CvBridge
from geometry_msgs.msg import TransformStamped
import numpy as np
import rclpy
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
import tf_transformations as transformations
from tf2_ros import TransformBroadcaster

from rt_gene.extract_landmarks_method_base import LandmarkMethodBase
import rt_gene.gaze_tools as gaze_tools
from rt_gene.kalman_stabilizer import Stabilizer
from rt_gene.tracker_face_encoding import FaceEncodingTracker
from rt_gene.tracker_sequential import SequentialTracker
from rt_gene_core import transforms as frame_transforms
from rt_gene_interfaces.msg import HeadPose, HeadPoseArray, Landmarks, LandmarksArray, SubjectImagesArray
from rt_gene_ros.subject_bridge import SubjectArrayBridge


class LandmarkNode(Node):
    def __init__(self):
        super().__init__("extract_landmarks")
        self._declare_parameters()

        self.landmark_method = LandmarkMethodBase(
            device_id_facedetection=self.get_parameter("device").value
        )
        self._apply_model_scale_from_parameters()

        if self.get_parameter("use_face_encoding_tracker").value:
            self.subject_tracker = FaceEncodingTracker(
                threshold=self.get_parameter("face_encoding_threshold").value
            )
        else:
            self.subject_tracker = SequentialTracker()

        self.bridge = CvBridge()
        self.subject_bridge = SubjectArrayBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_prefix = self.get_parameter("tf_prefix").value.strip("/") or "gaze"
        self.visualise_headpose = self.get_parameter("visualise_headpose").value
        self.pnp_iterate_after = self.get_parameter("pnp_iterate_after").value
        self.pose_stabilizers = {}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_frame = "camera_optical_frame"

        self.subject_pub = self.create_publisher(SubjectImagesArray, "subjects/images", qos_profile_sensor_data)
        self.headpose_pub = self.create_publisher(HeadPoseArray, "subjects/head_pose", QoSProfile(depth=10))
        self.landmark_pub = self.create_publisher(LandmarksArray, "subjects/landmarks", QoSProfile(depth=10))
        self.face_pub = self.create_publisher(Image, "subjects/head_pose_images", qos_profile_sensor_data)

        self.create_subscription(CameraInfo, "camera_info", self.camera_info_callback, qos_profile_sensor_data)
        self.create_subscription(Image, "image_raw", self.image_callback, qos_profile_sensor_data)
        self.add_on_set_parameters_callback(self.parameter_callback)

    def _declare_parameters(self):
        self.declare_parameter("device", "auto")
        self.declare_parameter("use_face_encoding_tracker", True)
        self.declare_parameter("face_encoding_threshold", 0.8)
        self.declare_parameter("tf_prefix", "gaze")
        self.declare_parameter("visualise_headpose", True)
        self.declare_parameter("pnp_iterate_after", False)
        self.declare_parameter("interpupillary_distance", 0.058)
        self.declare_parameter("model_size", 16.0)
        self.declare_parameter("head_pitch", 0.0)

    def _apply_model_scale_from_parameters(self):
        old_scale = self.landmark_method.model_size_rescale * self.landmark_method.interpupillary_distance
        self.landmark_method.model_points /= old_scale
        self.landmark_method.model_size_rescale = self.get_parameter("model_size").value
        self.landmark_method.interpupillary_distance = self.get_parameter("interpupillary_distance").value
        self.landmark_method.head_pitch = self.get_parameter("head_pitch").value
        self.landmark_method.model_points *= (
            self.landmark_method.model_size_rescale * self.landmark_method.interpupillary_distance
        )

    def parameter_callback(self, params):
        for param in params:
            if param.name in {"model_size", "interpupillary_distance"} and param.value <= 0:
                return SetParametersResult(successful=False, reason=f"{param.name} must be positive")
        for param in params:
            if param.name == "model_size":
                old_scale = self.landmark_method.model_size_rescale * self.landmark_method.interpupillary_distance
                self.landmark_method.model_points /= old_scale
                self.landmark_method.model_size_rescale = float(param.value)
                self.landmark_method.model_points *= (
                    self.landmark_method.model_size_rescale * self.landmark_method.interpupillary_distance
                )
            elif param.name == "interpupillary_distance":
                old_scale = self.landmark_method.model_size_rescale * self.landmark_method.interpupillary_distance
                self.landmark_method.model_points /= old_scale
                self.landmark_method.interpupillary_distance = float(param.value)
                self.landmark_method.model_points *= (
                    self.landmark_method.model_size_rescale * self.landmark_method.interpupillary_distance
                )
            elif param.name == "head_pitch":
                self.landmark_method.head_pitch = float(param.value)
        return SetParametersResult(successful=True)

    def camera_info_callback(self, msg):
        self.camera_matrix = np.array(msg.k, dtype=float).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=float)
        self.camera_frame = msg.header.frame_id.lstrip("/") or self.camera_frame
        if not np.any(self.camera_matrix):
            self.get_logger().error("Camera matrix is zero; publish calibrated camera_info before image_raw")
            self.camera_matrix = None

    def image_callback(self, msg):
        if self.camera_matrix is None:
            self.get_logger().warn("Waiting for camera_info", throttle_duration_sec=5.0)
            return

        color_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        self.update_subject_tracker(color_img)
        if not self.subject_tracker.get_tracked_elements():
            return

        self.subject_tracker.update_eye_images(self.landmark_method.eye_image_size)
        head_pose_images = []
        for subject_id, subject in self.subject_tracker.get_tracked_elements().items():
            if subject.left_eye_color is None or subject.right_eye_color is None:
                continue
            if subject_id not in self.pose_stabilizers:
                self.pose_stabilizers[subject_id] = [
                    Stabilizer(state_num=2, measure_num=1, cov_process=0.1, cov_measure=0.1)
                    for _ in range(6)
                ]

            success, head_rpy, translation = self.get_head_pose(subject.landmarks, subject_id)
            if not success:
                continue

            subject.head_rotation = head_rpy
            subject.head_translation = translation
            self.publish_pose(msg.header, translation, head_rpy, subject_id)

            if self.visualise_headpose:
                roll_pitch_yaw = list(
                    transformations.euler_from_matrix(
                        np.dot(frame_transforms.camera_to_ros, transformations.euler_matrix(*head_rpy))
                    )
                )
                face = cv2.resize(subject.face_color, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                head_pose_images.append(
                    LandmarkMethodBase.visualize_headpose_result(
                        face, gaze_tools.get_phi_theta_from_euler(gaze_tools.limit_yaw(roll_pitch_yaw))
                    )
                )

        subjects = self.subject_tracker.get_tracked_elements()
        self.publish_subjects(msg.header, subjects)
        if head_pose_images:
            image_msg = self.bridge.cv2_to_imgmsg(np.hstack(head_pose_images), "bgr8")
            image_msg.header = msg.header
            self.face_pub.publish(image_msg)

    def get_head_pose(self, landmarks, subject_id):
        try:
            success, rotation, translation, _ = cv2.solvePnPRansac(
                self.landmark_method.model_points,
                landmarks.reshape(len(self.landmark_method.model_points), 1, 2),
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.dist_coeffs,
                flags=cv2.SOLVEPNP_DLS,
            )
            if self.pnp_iterate_after:
                success, rotation, translation = cv2.solvePnP(
                    self.landmark_method.model_points,
                    landmarks.reshape(len(self.landmark_method.model_points), 1, 2),
                    rvec=rotation,
                    tvec=translation,
                    useExtrinsicGuess=True,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
        except cv2.error as exc:
            self.get_logger().warn(f"Could not estimate head pose: {exc}")
            return False, None, None

        if not success:
            return False, None, None

        rotation, translation = self.apply_kalman_filter(subject_id, rotation, translation / 1000.0)
        rotation[0] += self.landmark_method.head_pitch
        rotation_matrix, _ = cv2.Rodrigues(rotation)
        rotation_matrix = np.matmul(rotation_matrix, np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]))
        matrix = np.zeros((4, 4))
        matrix[:3, :3] = rotation_matrix
        matrix[3, 3] = 1
        return True, np.array(transformations.euler_from_matrix(matrix)), translation

    def apply_kalman_filter(self, subject_id, rotation, translation):
        stable_pose = []
        for value, stabilizer in zip(np.array((rotation, translation)).flatten(), self.pose_stabilizers[subject_id]):
            stabilizer.update([value])
            stable_pose.append(stabilizer.state[0])
        stable_pose = np.reshape(stable_pose, (-1, 3))
        return stable_pose[0], stable_pose[1]

    def publish_subjects(self, header, subjects):
        subject_msg = self.subject_bridge.images_to_msg(subjects, header)
        subject_msg.header.frame_id = self.camera_frame
        self.subject_pub.publish(subject_msg)

        landmarks_msg = LandmarksArray()
        landmarks_msg.header = subject_msg.header
        headpose_msg = HeadPoseArray()
        headpose_msg.header = subject_msg.header

        for subject_id, subject in subjects.items():
            if hasattr(subject, "landmarks"):
                item = Landmarks()
                item.subject_id = str(subject_id)
                item.landmarks = subject.landmarks.flatten().astype(float).tolist()
                landmarks_msg.subjects.append(item)
            if hasattr(subject, "head_rotation"):
                item = HeadPose()
                item.subject_id = str(subject_id)
                item.roll, item.pitch, item.yaw = [float(v) for v in subject.head_rotation]
                item.x, item.y, item.z = [float(v) for v in subject.head_translation]
                headpose_msg.subjects.append(item)

        self.landmark_pub.publish(landmarks_msg)
        self.headpose_pub.publish(headpose_msg)

    def publish_pose(self, header, translation, head_rpy, subject_id):
        msg = TransformStamped()
        msg.header = header
        msg.header.frame_id = self.camera_frame
        msg.child_frame_id = f"{self.tf_prefix}/head_pose/{subject_id}"
        msg.transform.translation.x = float(translation[0])
        msg.transform.translation.y = float(translation[1])
        msg.transform.translation.z = float(translation[2])
        q = transformations.quaternion_from_euler(*head_rpy)
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(msg)

    def update_subject_tracker(self, color_img):
        faceboxes = self.landmark_method.get_face_bb(color_img)
        if not faceboxes:
            self.subject_tracker.clear_elements()
            return
        self.subject_tracker.track(self.landmark_method.get_subjects_from_faceboxes(color_img, faceboxes))


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = LandmarkNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if node is not None:
                node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass
