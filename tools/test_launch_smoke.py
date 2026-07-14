import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2

from rt_gene_core.paths import demo_image_path


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def prepare_models():
    from rt_gene.download_tools import download_external_landmark_models, download_gaze_pytorch_models

    download_external_landmark_models()
    download_gaze_pytorch_models()


def ros_env(tmp):
    env = os.environ.copy()
    log_dir = Path(tmp) / "ros_logs"
    log_dir.mkdir()
    env["ROS_LOG_DIR"] = str(log_dir)
    env["ROS_DOMAIN_ID"] = str(1 + (os.getpid() + time.monotonic_ns()) % 100)
    return env


def stop_process(proc):
    try:
        os.killpg(proc.pid, signal.SIGINT)
    except (AttributeError, ProcessLookupError):
        proc.send_signal(signal.SIGINT)
    try:
        return proc.communicate(timeout=5)[0]
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (AttributeError, ProcessLookupError):
            proc.kill()
        return proc.communicate(timeout=5)[0]


def wait_for_topic(topic, proc, env, timeout=45):
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"launch exited early with {proc.returncode}")
        result = subprocess.run(
            [
                "ros2",
                "topic",
                "echo",
                "--no-daemon",
                "--spin-time",
                "5",
                "--once",
                "--timeout",
                "5",
                topic,
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=15,
            env=env,
        )
        last = result.stdout
        if result.returncode == 0 and last.strip():
            return last
        time.sleep(1.0)
    raise RuntimeError(f"No message received on {topic}:\n{last}")


def main():
    prepare_models()
    with tempfile.TemporaryDirectory() as tmp:
        env = ros_env(tmp)
        image = Path(tmp) / "launch_smoke.jpg"
        make_demo_image(image)
        command = [
            "ros2",
            "launch",
            "rt_gene_ros",
            "webcam_demo.launch.py",
            f"image_file:={image}",
            f"params_file:={REPO_ROOT / 'rt_gene_ros' / 'config' / 'params.yaml'}",
            "width:=640",
            "height:=480",
            "fps:=5.0",
            "visualise:=false",
        ]
        proc = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        topic_output = ""
        error = None
        try:
            topic_output = wait_for_topic("/subjects/gaze", proc, env)
        except Exception as exc:
            error = exc
        finally:
            output = stop_process(proc)
            scan_output = output + topic_output
        if error is not None:
            raise RuntimeError(f"{error}\nlaunch output:\n{output}")

    found = [pattern for pattern in BAD_LOG_PATTERNS if pattern in scan_output]
    if found:
        raise RuntimeError("Bad launch log patterns {}:\n{}".format(", ".join(found), output))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
