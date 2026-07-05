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
pixi run webcam-demo
```

Enable blink estimation too:

```bash
pixi run webcam-demo 0 "" "" 640 480 30.0 false 80 compressed auto cpu false true
```

Useful task arguments are positional. The full default-equivalent command is:

```bash
pixi run webcam-demo 0 "" "" 640 480 30.0 false 80 compressed auto cpu false false
```

Argument order is `camera_index`, `video_file`, `image_file`, `width`, `height`, `fps`, `loop`, `jpeg_quality`, `image_transport`, `device`, `landmark_device`, `visualise`, `blink`.
The demo defaults to `image_transport=compressed`, so the inference node subscribes to `image_raw/compressed`. Use `raw` in that position to subscribe directly to `image_raw`. The webcam launch defaults landmark/SFD work to CPU and leaves gaze on `device=auto`.

The default topics are relative and remappable:

- input: `image_raw`, `camera_info`
- outputs: `subjects/images`, `subjects/landmarks`, `subjects/head_pose`, `subjects/gaze`, `subjects/blink`
- visualisation images: `subjects/head_pose_images`, `subjects/gaze_images`, `subjects/blink_images`

On macOS, grant camera permission to the terminal or app that runs Pixi/Codex before using a real webcam. For a viewer with compression:

```bash
pixi run camera-view
```

Measure topic rates with:

```bash
pixi run topic-hz /subjects/gaze
pixi run demo-hz
pixi run demo-hz auto 30 compressed
```

`demo-hz` repeats the bundled demo image through `opencv_camera`, launches the ROS 2 pipeline, and measures `image_raw` and `subjects/gaze` with an in-process ROS subscriber. Add extra reliable topics with Pixi's passthrough separator, for example:

```bash
pixi run demo-hz -- --topic /image_raw/compressed
```

## Single Image Demo

Run the core pipeline on one image without starting ROS:

```bash
pixi run demo-image
pixi run demo-image /path/to/face.jpg
```

The first command uses a restored ROS 1 demo image bundled with `rt_gene_core`. It prints JSON with detected face boxes, head pose, translation, and gaze angles. The demo uses an approximate pinhole
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
pixi run test-ros-python
pixi run test-installed
pixi run test-camera
pixi run test-launch
```

`test-core` is hardware-free and network-free. `test-ros-python` covers ROS message conversion helpers such as timestamp propagation. `test-installed` verifies that Pixi imports the editable workspace sources instead of stale installed copies. `test-camera` uses a synthetic image to verify `image_raw`, `image_raw/compressed`, `camera_info`, and reliable image QoS. `test-launch` uses the bundled demo image and fails on startup tracebacks, process death, or QoS incompatibility warnings.

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
