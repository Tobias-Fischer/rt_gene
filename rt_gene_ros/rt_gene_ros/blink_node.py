import collections
import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import Image

from rt_bene.estimate_blink_pytorch import BlinkEstimatorPytorch
from rt_gene_interfaces.msg import Blink, BlinkArray, SubjectImagesArray
from rt_gene_ros.model_paths import resolve_model_files
from rt_gene_ros.subject_bridge import SubjectArrayBridge


DEFAULT_BLINK_MODELS = [
    "blink_model_pytorch_vgg16_allsubjects1.model",
    "blink_model_pytorch_vgg16_allsubjects2.model",
]


class BlinkNode(Node):
    def __init__(self):
        super().__init__("estimate_blink")
        self.declare_parameter("device", "auto")
        self.declare_parameter("model_files", DEFAULT_BLINK_MODELS)
        self.declare_parameter("model_type", "vgg16")
        self.declare_parameter("threshold", 0.425)
        self.declare_parameter("visualise", True)

        self.bridge = CvBridge()
        self.subject_bridge = SubjectArrayBridge()
        self.visualise = self.get_parameter("visualise").value
        self.estimator = BlinkEstimatorPytorch(
            device_id_blink=self.get_parameter("device").value,
            model_files=resolve_model_files(self.get_parameter("model_files").value, writable=True),
            model_type=self.get_parameter("model_type").value,
            threshold=self.get_parameter("threshold").value,
        )

        latest_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(SubjectImagesArray, "subjects/images", self.callback, latest_qos)
        self.blink_pub = self.create_publisher(BlinkArray, "subjects/blink", QoSProfile(depth=10))
        self.image_pub = self.create_publisher(Image, "subjects/blink_images", QoSProfile(depth=5))
        self.last_time = self.get_clock().now()
        self.freq = collections.deque(maxlen=30)
        self.latency = collections.deque(maxlen=30)

    def callback(self, msg):
        subjects = self.subject_bridge.msg_to_images(msg)
        left_eyes, right_eyes = [], []
        for subject in subjects.values():
            left, right = self.estimator.inputs_from_images(subject.left, subject.right)
            left_eyes.append(left)
            right_eyes.append(right)
        if not left_eyes:
            return

        probs = np.asarray(self.estimator.predict(left_eyes, right_eyes)).reshape(-1)
        self.publish_msg(msg.header, list(subjects.keys()), probs)

        if self.visualise:
            images = []
            for subject, probability in zip(subjects.values(), probs):
                face = cv2.resize(subject.face, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                images.append(self.estimator.overlay_prediction_over_img(face, probability))
            if images:
                image_msg = self.bridge.cv2_to_imgmsg(np.hstack(images), "bgr8")
                image_msg.header = msg.header
                self.image_pub.publish(image_msg)

        now = self.get_clock().now()
        stamp = Time.from_msg(msg.header.stamp)
        elapsed = (now - self.last_time).nanoseconds / 1e9
        if elapsed > 0:
            self.freq.append(1.0 / elapsed)
        self.latency.append((now - stamp).nanoseconds / 1e9)
        self.last_time = now

    def publish_msg(self, header, subject_ids, probabilities):
        msg = BlinkArray()
        msg.header = header
        for subject_id, probability in zip(subject_ids, probabilities):
            item = Blink()
            item.subject_id = str(subject_id)
            item.probability = float(probability)
            item.blink = bool(probability >= self.estimator.threshold)
            msg.subjects.append(item)
        self.blink_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BlinkNode()
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
