import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = [REPO_ROOT / "rt_gene_core" / "src", REPO_ROOT / "rt_gene_ros"]


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
)


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video: {path}")
    for index in range(20):
        frame = np.full((240, 320, 3), index * 10 % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def prepare_models():
    for path in reversed(SOURCE_PATHS):
        sys.path.insert(0, str(path))

    from rt_gene.download_tools import download_external_landmark_models, download_gaze_pytorch_models

    download_external_landmark_models()
    download_gaze_pytorch_models()


def ros_env(tmp):
    env = os.environ.copy()
    log_dir = Path(tmp) / "ros_logs"
    log_dir.mkdir()
    env["ROS_LOG_DIR"] = str(log_dir)
    env["ROS_DOMAIN_ID"] = str(30 + os.getpid() % 50)
    pythonpath = os.pathsep.join(str(path) for path in SOURCE_PATHS)
    if env.get("PYTHONPATH"):
        pythonpath = os.pathsep.join([pythonpath, env["PYTHONPATH"]])
    env["PYTHONPATH"] = pythonpath
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


def as_text(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def main():
    prepare_models()
    with tempfile.TemporaryDirectory() as tmp:
        env = ros_env(tmp)
        video = Path(tmp) / "launch_smoke.avi"
        make_video(video)
        command = [
            "ros2",
            "launch",
            "rt_gene_ros",
            "webcam_demo.launch.py",
            f"video_file:={video}",
            f"params_file:={REPO_ROOT / 'rt_gene_ros' / 'config' / 'params.yaml'}",
            "loop:=true",
            "width:=0",
            "height:=0",
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
        try:
            output, _ = proc.communicate(timeout=45)
            if proc.returncode != 0:
                raise RuntimeError(output)
        except subprocess.TimeoutExpired as exc:
            output = as_text(exc.output) + as_text(stop_process(proc))

    found = [pattern for pattern in BAD_LOG_PATTERNS if pattern in output]
    if found:
        raise RuntimeError("Bad launch log patterns {}:\n{}".format(", ".join(found), output))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
