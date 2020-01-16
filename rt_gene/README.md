# RT-GENE: Real-Time Eye Gaze and Blink Estimation in Natural Environments
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![HitCount](http://hits.dwyl.io/Tobias-Fischer/rt_gene.svg)
![stars](https://img.shields.io/github/stars/Tobias-Fischer/rt_gene.svg?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Tobias-Fischer/rt_gene.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/Tobias-Fischer/rt_gene.svg?style=flat-square)


## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (note that some libraries are used that are distributed under a different license, see [below](./README.md#list-of-libraries)). Commercial usage is not permitted. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html):

```
@inproceedings{FischerECCV2018,
author = {Tobias Fischer and Hyung Jin Chang and Yiannis Demiris},
title = {{RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments}},
booktitle = {European Conference on Computer Vision},
year = {2018},
month = {September},
pages = {339--357}
}
```

RT-GENE was supported in part by the Samsung Global Research Outreach program, and in part by the EU Horizon 2020 Project PAL (643783-RIA).

If you use our blink estimation code, please also cite the relevant [paper](http://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Cortacero_RT-BENE_A_Dataset_and_Baselines_for_Real-Time_Blink_Estimation_in_ICCVW_2019_paper.html):
```
@inproceedings{CortaceroICCV2019W,
author={Kevin Cortacero and Tobias Fischer and Yiannis Demiris},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision Workshops},
title = {RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments},
year = {2019},
}
```

RT-BENE was supported by the EU Horizon 2020 Project PAL (643783-RIA) and a Royal Academy of Engineering Chair in Emerging Technologies to Yiannis Demiris.

More information can be found on the Personal Robotic Lab's website: <https://www.imperial.ac.uk/personal-robotics/software/>.

## Requirements
### Manual installation
1. Download, install, and configure ROS (full installation; we recommend the Kinectic or Melodic distributions of ROS depending on your Ubuntu version): http://wiki.ros.org/kinetic/Installation or http://wiki.ros.org/melodic/Installation
1. Install additional packages for ROS:
    - For kinetic: `sudo apt-get install python-catkin-tools ros-kinetic-ros-numpy ros-kinetic-camera-info-manager-py ros-kinetic-uvc-camera libcamera-info-manager-dev`
    - For melodic: `sudo apt-get install python-catkin-tools python-catkin-pkg ros-melodic-uvc-camera libcamera-info-manager-dev`
1. Install required Python packages:
    - For `conda` users (recommended): `conda install tensorflow-gpu numpy scipy tqdm pytorch torchvision pillow && conda install -c conda-forge dlib && pip install empy rospkg`
    - For `pip` users: `pip install tensorflow-gpu numpy scipy tqdm torch torchvision Pillow dlib opencv-python`
1. Download and build RT-GENE:
    1. `cd $HOME/catkin_ws/src && git clone https://github.com/Tobias-Fischer/rt_gene.git`
    1. `cd $HOME/catkin_ws && catkin build`

### Optional ensemble model files
- To use an ensemble scheme using 4 models trained on the MPII, UTMV and RT-GENE datasets, you need to download the additional model files (make sure you comply with the licenses of [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) and [UTMV](http://www.hci.iis.u-tokyo.ac.jp/datasets/)! these model files are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)).
    - `cd $HOME/catkin_ws/src && python ./rt_gene/rt_gene/scripts/download_models.py`
- Finally, open `$(rospack find rt_gene)/launch/estimate_gaze.launch` and comment out `<rosparam param="model_files">['model_nets/Model_allsubjects1.h5']</rosparam>` and uncomment `<!--rosparam param="model_files">['model_nets/all_subjects_mpii_prl_utmv_0_02.h5', ..., ..., ...</rosparam-->`


### Requirements for live gaze estimation (Kinect One)
- Follow instructions for https://github.com/code-iai/iai_kinect2
- Make sure the calibration is saved correctly (https://github.com/code-iai/iai_kinect2/tree/master/kinect2_calibration#calibrating-the-kinect-one)

### Requirements for live gaze estimation (webcam / RGB only)
- Calibrate your camera (http://wiki.ros.org/camera_calibration). 
- Save the resulting `*.yaml` file to `$(rospack find rt_gene)/webcam_configs/`.
- Change the entry for the `camera_info_url` in the `$(rospack find rt_gene)/launch/start_webcam.launch` file.

## Instructions for estimating gaze

### Estimate gaze live from Kinect One
1) `roscore`
1) `roslaunch rt_gene start_kinect.launch`
1) `roslaunch rt_gene estimate_gaze.launch`

### Estimate gaze live from Webcam / RGB only camera
1) `roscore`
1) `roslaunch rt_gene start_webcam.launch`
1) `roslaunch rt_gene estimate_gaze.launch`

### Estimate gaze from Video
1) `roscore`
1) `roslaunch rt_gene start_video.launch` (make sure to change the `camera_info_url` and `video_file` arguments)
1) `roslaunch rt_gene estimate_gaze.launch`

### Estimate gaze from ROSBag
1) `roscore`
1) `roslaunch rt_gene start_rosbag.launch rosbag_file:=/path/to/rosbag.bag` (this assumes a recording with the Kinect v2 and might need adjustments)
1) `roslaunch rt_gene estimate_gaze.launch ros_frame:=kinect2_nonrotated_link`

## Instructions for estimating blinks
Follow the instructions for estimating gaze above, and run in addition `roslaunch rt_gene estimate_blink.launch`. Note that the blink estimation relies on the `extract_landmarks_node.py` node, however can run independently from the `estimate_gaze.py` node.

## List of libraries
- ROS; [BSD 3-clause](https://opensource.org/licenses/BSD-3-Clause), [Link to website](http://ros.org/)
- Tensorflow; [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), [Link to website](http://tensorflow.org/)
- 3DDFA face alignment in [./src/rt_gene/ThreeDDFA](./src/rt_gene/ThreeDDFA); [MIT License](https://opensource.org/licenses/MIT), [Link to GitHub](https://github.com/cleardusk/3DDFA), [Link to paper](https://arxiv.org/abs/1804.01005)
- S3FD face detector in [./src/rt_gene/SFD](./src/rt_gene/SFD); [BSD 3-clause](https://opensource.org/licenses/BSD-3-Clause), [Link to GitHub](https://github.com/1adrianb/face-alignment)
- Kalman filter in [./src/rt_gene/kalman_stabilizer.py](./src/rt_gene/kalman_stabilizer.py): [MIT License](https://opensource.org/licenses/MIT), [Link to GitHub](https://github.com/yinguobing/head-pose-estimation)

