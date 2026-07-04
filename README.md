# RT-GENE / RT-BENE ROS 2

RT-GENE estimates gaze direction from face and eye images; RT-BENE estimates blinks. This branch is the modern ROS 2 Jazzy runtime port. It uses Pixi with RoboStack, keeps the runtime in PyTorch, and is intentionally not API-compatible with the old ROS 1 package.

The archived ROS 1 code, TensorFlow-era code, old standalone scripts, training folders, and historical assets live on the [`ros1`](https://github.com/Tobias-Fischer/rt_gene/tree/ros1) branch.

## Packages

- `rt_gene_core`: pure Python runtime code for landmarks, gaze, blink, model downloads, and PyTorch device selection. It has no ROS imports.
- `rt_gene_interfaces`: ROS 2 messages for subject images, landmarks, head pose, gaze, and blink results.
- `rt_gene_ros`: `rclpy` nodes and launch files.
- `opencv_camera`: independent OpenCV webcam/video publisher for `sensor_msgs/Image` and `sensor_msgs/CameraInfo`.

## Install

Pixi builds the ROS packages as conda packages with `pixi-build-ros`.

```bash
pixi install
```

Supported workspace platforms are `osx-arm64`, `linux-64`, and `win-64`. macOS and Linux are the intended targets; Windows is best-effort and currently untested.

## Webcam Demo

Run the camera alone:

```bash
pixi run ros2 run opencv_camera camera_node
```

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
  device:=auto \
  visualise:=true
```

The default topics are relative and remappable:

- input: `image_raw`, `camera_info`
- outputs: `subjects/images`, `subjects/landmarks`, `subjects/head_pose`, `subjects/gaze`, `subjects/blink`
- visualisation images: `subjects/head_pose_images`, `subjects/gaze_images`, `subjects/blink_images`

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
```

The current tests cover PyTorch device resolution, OpenCV calibration YAML loading, and a guard that active packages do not reintroduce TensorFlow imports.

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
