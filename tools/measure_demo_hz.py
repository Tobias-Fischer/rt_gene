import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

from rt_gene_core.paths import demo_image_path


BAD_LOG_PATTERNS = (
    "Traceback",
    "process has died",
    "uncaught exception",
    "NSInternalInconsistencyException",
    "API misuse: setting the main menu on a non-main thread",
    "RELIABILITY_QOS_POLICY",
    "Abort trap",
    "Expected one of cpu",
    "KeyError: 'content-length'",
    "KeyError: 'Content-length'",
    "Content-length",
    "PosixPath",
    "has no attribute 'rfind'",
    "pretrained' is deprecated",
    "Arguments other than a weight enum",
    "requesting incompatible QoS",
)


def make_demo_image(path, width=640, height=480):
    image = cv2.imread(str(demo_image_path()), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read demo image: {demo_image_path()}")

    center_x, center_y = int(image.shape[1] * 0.49), int(image.shape[0] * 0.38)
    left = max(0, min(center_x - width // 2, image.shape[1] - width))
    top = max(0, min(center_y - height // 2, image.shape[0] - height))
    frame = image[top:top + height, left:left + width]
    if frame.shape[:2] != (height, width):
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Could not create demo image: {path}")


def stop_process(proc):
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return ""
    try:
        return proc.communicate(timeout=5)[0]
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        return proc.communicate(timeout=5)[0]


def topic_type(topic, env, timeout):
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["ros2", "topic", "type", "--no-daemon", topic],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=10,
            env=env,
        )
        last = result.stdout
        if result.returncode == 0 and last.strip():
            return last.strip()
        time.sleep(1.0)
    raise RuntimeError(f"Topic {topic} did not appear:\n{last}")


def measure_hz(topic, env, timeout):
    msg_type = topic_type(topic, env, timeout)
    os.environ["ROS_DOMAIN_ID"] = env["ROS_DOMAIN_ID"]
    os.environ["ROS_LOG_DIR"] = env["ROS_LOG_DIR"]

    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rosidl_runtime_py.utilities import get_message

    rclpy.init(args=None)
    node = rclpy.create_node(f"measure_hz_{topic.strip('/').replace('/', '_')}")
    times = []
    qos = QoSProfile(depth=10)
    if topic == "/subjects/images":
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
    node.create_subscription(get_message(msg_type), topic, lambda _msg: times.append(time.monotonic()), qos)
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline and len(times) < 6:
            rclpy.spin_once(node, timeout_sec=0.1)
        if len(times) < 2:
            raise RuntimeError(f"No hz result for {topic}: received {len(times)} message(s)")
        return f"average rate: {(len(times) - 1) / (times[-1] - times[0]):.3f}"
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Launch the bundled-image ROS demo and measure topic rates.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--landmark-device", default="cpu")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--image-transport", choices=("raw", "compressed"), default="compressed")
    parser.add_argument("--topic", action="append")
    args = parser.parse_args()
    topics = args.topic or ["/image_raw", "/subjects/gaze"]

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        image = tmp_path / "rt_gene_demo.jpg"
        make_demo_image(image)
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(1 + (os.getpid() + time.monotonic_ns()) % 100)
        env["ROS_LOG_DIR"] = str(tmp_path / "ros_logs")
        Path(env["ROS_LOG_DIR"]).mkdir()

        launch = subprocess.Popen(
            [
                "ros2",
                "launch",
                "rt_gene_ros",
                "webcam_demo.launch.py",
                f"image_file:={image}",
                "width:=640",
                "height:=480",
                "fps:=30.0",
                f"device:={args.device}",
                f"landmark_device:={args.landmark_device}",
                f"image_transport:={args.image_transport}",
                "visualise:=false",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            for topic in topics:
                print(f"{topic}: {measure_hz(topic, env, args.timeout)}")
        finally:
            output = stop_process(launch)
            found = [pattern for pattern in BAD_LOG_PATTERNS if pattern in output]
            if found:
                print(output, file=sys.stderr)
                raise RuntimeError(f"Bad launch log patterns: {', '.join(found)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
