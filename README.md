# RT-GENE & RT-BENE: Real-Time Eye Gaze And Blink Estimation In Natural Environments

This repository contains code and dataset references for [RT-GENE (gaze estimation, ECCV 2018)](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html) and [RT-BENE (blink estimation, ICCV 2019 Workshops)](http://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Cortacero_RT-BENE_A_Dataset_and_Baselines_for_Real-Time_Blink_Estimation_in_ICCVW_2019_paper.html).

## RT-GENE Paper, Dataset, And Citation

RT-GENE estimates gaze direction from face and eye images in natural environments. The RT-GENE code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); commercial usage is not permitted.

The accompanying RT-GENE dataset is available on [Zenodo](https://zenodo.org/record/2529036) with an [alternative link](https://goo.gl/tfUaDm). For more datasets and open-source software, see the Personal Robotics Lab software page: <https://www.imperial.ac.uk/personal-robotics/software/>.

Acknowledgements: RT-GENE was supported in part by the Samsung Global Research Outreach program, and in part by the EU Horizon 2020 Project PAL (643783-RIA).

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

## RT-BENE Paper, Dataset, And Citation

RT-BENE estimates blinks from eye image patches. The RT-BENE code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/); commercial usage is not permitted.

The [RT-BENE dataset on Zenodo](https://zenodo.org/record/3685316) contains eye image patches and associated blink annotations. It was manually annotated from the "noglasses" part of the RT-GENE dataset.

Acknowledgements: RT-BENE was supported by the EU Horizon 2020 Project PAL (643783-RIA) and a Royal Academy of Engineering Chair in Emerging Technologies to Yiannis Demiris.

If you use RT-BENE, please cite:

```bibtex
@inproceedings{CortaceroICCV2019W,
  author = {Kevin Cortacero and Tobias Fischer and Yiannis Demiris},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision Workshops},
  title = {RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments},
  year = {2019}
}
```

## Modern ROS 2 Branch

This branch is the modern ROS 2 runtime port. It uses Pixi with RoboStack, keeps the runtime in PyTorch, and is intentionally not API-compatible with the old ROS 1 package.

The archived ROS 1 code, TensorFlow-era code, old standalone subprojects, training folders, and historical assets live on the [`ros1`](https://github.com/Tobias-Fischer/rt_gene/tree/ros1) branch.

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

## Standalone Demos

These demos run the core PyTorch/OpenCV code without starting ROS 2:

```bash
pixi run demo-image
pixi run demo-image /path/to/face.jpg
pixi run demo-blink
pixi run demo-blink /path/to/face.jpg
```

`demo-image` runs RT-GENE gaze and head-pose estimation on one face image. `demo-blink` runs RT-BENE blink estimation on one face image by detecting the face, cropping eye patches, and estimating blink probability. Both default to the bundled restored ROS 1 gaze sample image.

For the old RT-BENE eye-patch workflow, provide explicit eye crops:

```bash
pixi run python -m rt_bene.single_image_demo --left-eye /path/to/left.png --right-eye /path/to/right.png
```

The editable package also installs ROS-free console scripts:

```bash
pixi run rt-gene-demo-image
pixi run rt-bene-demo-blink
```

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

Blink estimation is wrapped as the `rt_gene_ros` `estimate_blink` executable. It subscribes to `subjects/images` from `extract_landmarks` and publishes `subjects/blink`; with visualisation enabled it also publishes `subjects/blink_images`.

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
