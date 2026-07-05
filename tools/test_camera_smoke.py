import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def make_video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (320, 240))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create test video: {path}")
    for index in range(30):
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = (index * 7) % 255
        frame[:, :, 1] = np.arange(320, dtype=np.uint8)
        frame[:, :, 2] = np.arange(240, dtype=np.uint8)[:, None]
        writer.write(frame)
    writer.release()


def ros_env(tmp):
    env = os.environ.copy()
    log_dir = Path(tmp) / "ros_logs"
    log_dir.mkdir()
    env["ROS_LOG_DIR"] = str(log_dir)
    env["ROS_DOMAIN_ID"] = str(100 + (os.getpid() + time.monotonic_ns()) % 120)
    return env


def run(command, env, timeout=8):
    return subprocess.run(
        command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout, env=env
    )


def wait_for(command, proc, env, timeout=10):
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            try:
                output = proc.communicate(timeout=1)[0]
            except subprocess.TimeoutExpired:
                output = "<camera_node exited but output could not be read>"
            raise RuntimeError(f"camera_node exited early with {proc.returncode}\n{output}")
        result = run(command, env, timeout=8)
        last = result.stdout
        if result.returncode == 0:
            return last
        time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {' '.join(command)}\n{last}")


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


def publisher_qos_blocks(topic_info):
    blocks = []
    current = []
    for line in topic_info.splitlines():
        if line.startswith("Node name: "):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return [block for block in blocks if "Endpoint type: PUBLISHER" in block]


def assert_reliable_publishers(topic_info):
    publishers = publisher_qos_blocks(topic_info)
    if not publishers:
        raise RuntimeError(f"/image_raw has no publishers:\n{topic_info}")
    best_effort = [block for block in publishers if "Reliability: BEST_EFFORT" in block]
    unreliable = [block for block in publishers if "Reliability: RELIABLE" not in block]
    if best_effort or unreliable:
        raise RuntimeError(f"/image_raw publishers are not reliable by default:\n{topic_info}")


def scalar_int(output):
    for line in output.splitlines():
        line = line.strip()
        if line:
            return int(line)
    raise RuntimeError(f"Expected integer output, got:\n{output}")


def main():
    with tempfile.TemporaryDirectory() as tmp:
        env = ros_env(tmp)
        video = Path(tmp) / "camera_smoke.avi"
        make_video(video)
        proc = subprocess.Popen(
            [
                "ros2",
                "run",
                "opencv_camera",
                "camera_node",
                "--ros-args",
                "-p",
                f"video_file:={video}",
                "-p",
                "loop:=true",
                "-p",
                "width:=160",
                "-p",
                "height:=120",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        error = None
        try:
            camera_info_width = wait_for(
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
                    "--field",
                    "width",
                    "/camera_info",
                ],
                proc,
                env,
            )
            if scalar_int(camera_info_width) != 160:
                raise RuntimeError(f"/camera_info width does not match requested output size:\n{camera_info_width}")
            image_width = wait_for(
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
                    "--field",
                    "width",
                    "/image_raw",
                ],
                proc,
                env,
            )
            if scalar_int(image_width) != 160:
                raise RuntimeError(f"/image_raw width does not match requested output size:\n{image_width}")
            wait_for(
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
                    "--field",
                    "format",
                    "/image_raw/compressed",
                ],
                proc,
                env,
            )
            info = wait_for(["ros2", "topic", "info", "--no-daemon", "--spin-time", "5", "-v", "/image_raw"], proc, env)
            assert_reliable_publishers(info)
        except Exception as exc:
            error = exc
        finally:
            output = stop_process(proc)
        if error is not None:
            raise RuntimeError(f"{error}\ncamera_node output:\n{output}")
        bad = (
            "Traceback",
            "process has died",
            "RELIABILITY_QOS_POLICY",
            "Abort trap",
            "uncaught exception",
            "NSInternalInconsistencyException",
        )
        if any(pattern in output for pattern in bad):
            raise RuntimeError(output)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)
