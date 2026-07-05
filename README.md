# RT-GENE / RT-BENE ROS 2

RT-GENE estimates gaze direction from face and eye images; RT-BENE estimates blinks. This branch is the modern ROS 2 Jazzy runtime port. It uses Pixi with RoboStack, keeps the runtime in PyTorch, and is intentionally not API-compatible with the old ROS 1 package.

The archived ROS 1 code, TensorFlow-era code, old standalone scripts, training folders, and historical assets live on the [`ros1`](https://github.com/Tobias-Fischer/rt_gene/tree/ros1) branch.

## Packages

- `rt_gene_core`: pure Python runtime code for landmarks, gaze, blink, model downloads, and PyTorch device selection. It has no ROS imports.
- `rt_gene_interfaces`: ROS 2 messages for subject images, landmarks, head pose, gaze, and blink results.
- `rt_gene_ros`: `rclpy` nodes and launch files.
- `opencv_camera`: independent OpenCV webcam/video publisher for `sensor_msgs/Image`, `sensor_msgs/CompressedImage`, and `sensor_msgs/CameraInfo`.

## Install

Pixi builds the ROS packages as conda packages with `pixi-build-ros`.

```bash
pixi install
```

`rt_gene_core` and `rt_gene_ros` are layered in as editable Python packages, so Python source edits are picked up by `pixi run` without a rebuild. Re-run `pixi install` after changing Pixi manifests, messages, or C++ packages.

Supported workspace platforms are `osx-arm64`, `linux-64`, and `win-64`. macOS and Linux are the intended targets; Windows is best-effort and currently untested.

## Webcam Demo

Run the camera alone:

```bash
pixi run ros2 run opencv_camera camera_node
```

The camera publishes raw images on `image_raw`, JPEG-compressed images on `image_raw/compressed`, and calibration on `camera_info`. Raw image QoS defaults to reliable so RViz can subscribe without a reliability warning.

Run the full webcam gaze demo:

```bash
pixi run ros2 launch rt_gene_ros webcam_demo.launch.py
```

Enable blink estimation too:

```bash
pixi run ros2 launch rt_gene_ros webcam_demo.launch.py blink:=true
```

Useful launch arguments:

```bash
pixi run ros2 launch rt_gene_ros webcam_demo.launch.py \
  camera_index:=0 \
  calibration_file:=/path/to/calibration.yaml \
  width:=640 \
  height:=480 \
  fps:=30 \
  jpeg_quality:=80 \
  device:=auto \
  visualise:=true
```

The default topics are relative and remappable:

- input: `image_raw`, `camera_info`
- outputs: `subjects/images`, `subjects/landmarks`, `subjects/head_pose`, `subjects/gaze`, `subjects/blink`
- visualisation images: `subjects/head_pose_images`, `subjects/gaze_images`, `subjects/blink_images`

On macOS, grant camera permission to the terminal or app that runs Pixi/Codex before using a real webcam. For a viewer with compression:

```bash
pixi run ros2 run image_view image_view --ros-args -r image:=/image_raw -p image_transport:=compressed
```

## Single Image Demo

Run the core pipeline on one image without starting ROS:

```bash
pixi run python -m rt_gene.single_image_demo /path/to/face.jpg --device auto
```

It prints JSON with detected face boxes, head pose, translation, and gaze angles. The demo uses an approximate pinhole
camera model by default; pass `--focal-length-px` if you know the focal length for the image.

## Model Files

Small bundled landmark assets are installed with `rt_gene_core`. Larger PyTorch gaze and blink models are downloaded on demand into:

```text
~/.cache/rt_gene/model_nets
```

Set `RT_GENE_MODEL_DIR` to use another model directory.

## Device Selection

All runtime nodes accept a `device` parameter:

- `auto`: choose `mps` when available, otherwise `cuda`, otherwise `cpu`
- `mps`: Apple Metal backend
- `cuda` or `cuda:0`: CUDA backend
- `cpu`: CPU backend

Example:

```bash
pixi run ros2 launch rt_gene_ros webcam_demo.launch.py device:=mps
```

## Interfaces

Inspect the modern ROS 2 messages:

```bash
pixi run ros2 interface show rt_gene_interfaces/msg/GazeArray
```

## Tests

```bash
pixi run test-core
pixi run test-installed
pixi run test-camera
pixi run test-launch
```

`test-core` is hardware-free and network-free. `test-installed` verifies that Pixi imports the editable workspace sources instead of stale installed copies. `test-camera` uses a synthetic video to verify `image_raw`, `image_raw/compressed`, `camera_info`, and reliable image QoS. `test-launch` uses a synthetic video and fails on startup tracebacks, process death, or QoS incompatibility warnings.

## Citation

If you use RT-GENE, please cite:

```bibtex
@inproceedings{FischerECCV2018,
  author = {Tobias Fischer and Hyung Jin Chang and Yiannis Demiris},
  title = {{RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments}},
  booktitle = {European Conference on Computer Vision},
  year = {2018},
  month = {September},
  pages = {339--357}
}
```

If you use RT-BENE, please cite:

```bibtex
@inproceedings{CortaceroICCV2019W,
  author = {Kevin Cortacero and Tobias Fischer and Yiannis Demiris},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision Workshops},
  title = {RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments},
  year = {2019}
}
```

The code is licensed under CC BY-NC-SA 4.0; commercial usage is not permitted.
