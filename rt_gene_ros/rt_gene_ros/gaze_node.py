import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image
import tf_transformations as transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped

import rt_gene.gaze_tools as gaze_tools
from rt_gene.estimate_gaze_pytorch import GazeEstimator
from rt_gene_core import transforms as frame_transforms
from rt_gene_interfaces.msg import Gaze, GazeArray, SubjectImagesArray
from rt_gene_ros.model_paths import resolve_model_files
from rt_gene_ros.subject_bridge import SubjectArrayBridge
from cv_bridge import CvBridge


DEFAULT_GAZE_MODELS = [
    "gaze_model_pytorch_vgg16_prl_mpii_allsubjects1.model",
    "gaze_model_pytorch_vgg16_prl_mpii_allsubjects2.model",
    "gaze_model_pytorch_vgg16_prl_mpii_allsubjects3.model",
    "gaze_model_pytorch_vgg16_prl_mpii_allsubjects4.model",
]


class GazeNode(Node):
    def __init__(self):
        super().__init__("estimate_gaze")
        self.declare_parameter("device", "auto")
        self.declare_parameter("tf_prefix", "gaze")
        self.declare_parameter("visualise_eyepose", True)
        self.declare_parameter("model_files", DEFAULT_GAZE_MODELS)

        self.bridge = CvBridge()
        self.subject_bridge = SubjectArrayBridge()
        self.tf_prefix = self.get_parameter("tf_prefix").value.strip("/") or "gaze"
        self.visualise_eyepose = self.get_parameter("visualise_eyepose").value
        self.estimator = GazeEstimator(
            self.get_parameter("device").value,
            resolve_model_files(self.get_parameter("model_files").value, writable=True),
        )

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(SubjectImagesArray, "subjects/images", self.image_callback, qos_profile_sensor_data)
        self.gaze_pub = self.create_publisher(GazeArray, "subjects/gaze", QoSProfile(depth=10))
        self.image_pub = self.create_publisher(Image, "subjects/gaze_images", QoSProfile(depth=5))
        self.last_time = self.get_clock().now()
        self.freq = collections.deque(maxlen=30)
        self.latency = collections.deque(maxlen=30)

    def image_callback(self, msg):
        subjects = self.subject_bridge.msg_to_images(msg)
        stamp = Time.from_msg(msg.header.stamp)
        input_r, input_l, input_head, valid_subjects = [], [], [], []

        for subject_id, subject in subjects.items():
            try:
                transform = self.tf_buffer.lookup_transform(
                    msg.header.frame_id,
                    f"{self.tf_prefix}/head_pose/{subject_id}",
                    stamp,
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
                tf2_ros.TransformException,
            ):
                continue

            rot = transform.transform.rotation
            matrix = transformations.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
            euler = list(transformations.euler_from_matrix(np.dot(frame_transforms.camera_to_ros, matrix)))
            phi_head, theta_head = gaze_tools.get_phi_theta_from_euler(gaze_tools.limit_yaw(euler))
            input_head.append([theta_head, phi_head])
            input_r.append(self.estimator.input_from_image(subject.right))
            input_l.append(self.estimator.input_from_image(subject.left))
            valid_subjects.append(subject_id)

        if not valid_subjects:
            return

        estimates = self.estimator.estimate_gaze_twoeyes(input_l, input_r, input_head)
        self.publish_gaze_msg(msg.header, valid_subjects, estimates.tolist())

        gaze_images = []
        for subject_id, gaze in zip(valid_subjects, estimates.tolist()):
            self.publish_gaze_tf(gaze, msg.header, subject_id)
            if self.visualise_eyepose:
                subject = subjects[subject_id]
                gaze_images.append(
                    np.concatenate(
                        (
                            self.estimator.visualize_eye_result(subject.right, gaze),
                            self.estimator.visualize_eye_result(subject.left, gaze),
                        ),
                        axis=1,
                    )
                )

        if gaze_images:
            image_msg = self.bridge.cv2_to_imgmsg(np.hstack(gaze_images).astype(np.uint8), "bgr8")
            image_msg.header = msg.header
            self.image_pub.publish(image_msg)

        now = self.get_clock().now()
        elapsed = (now - self.last_time).nanoseconds / 1e9
        if elapsed > 0:
            self.freq.append(1.0 / elapsed)
        self.latency.append((now - stamp).nanoseconds / 1e9)
        self.last_time = now

    def publish_gaze_msg(self, header, subject_ids, gazes):
        msg = GazeArray()
        msg.header = header
        for subject_id, gaze in zip(subject_ids, gazes):
            item = Gaze()
            item.subject_id = str(subject_id)
            item.theta = float(gaze[0])
            item.phi = float(gaze[1])
            msg.subjects.append(item)
        self.gaze_pub.publish(msg)

    def publish_gaze_tf(self, gaze, header, subject_id):
        theta, phi = gaze
        q = transformations.quaternion_from_euler(*gaze_tools.get_euler_from_phi_theta(phi, theta))
        msg = TransformStamped()
        msg.header = header
        msg.header.frame_id = f"{self.tf_prefix}/head_pose/{subject_id}"
        msg.child_frame_id = f"{self.tf_prefix}/gaze/{subject_id}"
        msg.transform.translation.z = 0.05
        msg.transform.rotation.x = float(q[0])
        msg.transform.rotation.y = float(q[1])
        msg.transform.rotation.z = float(q[2])
        msg.transform.rotation.w = float(q[3])
        self.tf_broadcaster.sendTransform(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GazeNode()
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
