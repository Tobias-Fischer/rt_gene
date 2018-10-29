# License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html):

```
@inproceedings{FischerECCV2018,
author = {Tobias Fischer and Hyung Jin Chang and Yiannis Demiris},
title = {{RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments}},
booktitle = {European Conference on Computer Vision},
year = {2018},
month = {September},
pages = {334--352}
}
```

This work was supported in part by the Samsung Global Research Outreach program, and in part by the EU Horizon 2020 Project PAL (643783-RIA).

More information can be found on the Personal Robotic Lab's website: <https://www.imperial.ac.uk/personal-robotics/software/>.

# Requirements
## Manual installation
1) Download, install, and configure ROS (full installation; we recommend the Kinectic distribution of ROS): http://wiki.ros.org/kinetic/Installation
1) `sudo apt-get install python-catkin-tools ros-$ROS_DISTRO-ros-numpy ros-$ROS_DISTRO-camera-info-manager-py ros-$ROS_DISTRO-uvc-camera`
1) `pip install tensorflow-gpu keras numpy scipy tqdm torch torchvision`
1) `cd $HOME/catkin_ws/src && git clone https://github.com/Tobias-Fischer/rt_gene.git`
1) `cd $HOME/catkin_ws && catkin build`

## Optional ensemble model files
- To use an ensemble scheme using 4 models trained on the MPII, UTMV and RT-GENE datasets, download the following files (make sure you comply with the licenses of [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) and [UTMV](http://www.hci.iis.u-tokyo.ac.jp/datasets/)! these model files are licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)):
    - `wget https://imperialcollegelondon.box.com/shared/static/5cjnijpo8qxawbkik0gjrmyc802j2h1v.h5 -O $(rospack find rt_gene)/model_nets/all_subjects_mpii_prl_utmv_0_02`
    - `wget https://imperialcollegelondon.box.com/shared/static/1ye5jlh5ce11f93yn1s36uysjta7a3ob.h5 -O $(rospack find rt_gene)/model_nets/all_subjects_mpii_prl_utmv_1_02`
    - `wget https://imperialcollegelondon.box.com/shared/static/5vl9samndju9zhygtai8z6kkpw2jmjll.h5 -O $(rospack find rt_gene)/model_nets/all_subjects_mpii_prl_utmv_2_02`
    - `wget https://imperialcollegelondon.box.com/shared/static/hmcoxopu4xetic5bm47xqrl5mqktpg92.h5 -O $(rospack find rt_gene)/model_nets/all_subjects_mpii_prl_utmv_3_02`
- Finally, open `$(rospack find rt_gene)/launch/estimate_gaze.launch` and comment out `<rosparam param="model_files">['model_nets/Model_allsubjects1.h5']</rosparam>` and uncomment `<!--rosparam param="model_files">['model_nets/all_subjects_mpii_prl_utmv_0_02.h5', ..., ..., ...</rosparam-->`

## Improved head pose estimation
This is currently disabled by default as the DNN module in the OpenCV shipped by ROS is broken (see https://github.com/ros-gbp/opencv3-release/issues/20). To enable, build OpenCV3 from sources and link the ROS OpenCV to the one compiled from sources. Then download the following files:
- `wget https://github.com/yinguobing/head-pose-estimation/raw/master/assets/res10_300x300_ssd_iter_140000.caffemodel -P $(rospack find rt_gene)/model_nets`
- `wget https://raw.githubusercontent.com/yinguobing/head-pose-estimation/master/assets/deploy.prototxt -P $(rospack find rt_gene)/model_nets`

Finally, set `use_mtcnn` to `False` in `$(rospack find rt_gene)/launch/estimate_gaze.launch`

## Requirements for live gaze estimation (Kinect One)
- Follow instructions for https://github.com/code-iai/iai_kinect2
- Make sure the calibration is saved correctly (https://github.com/code-iai/iai_kinect2/tree/master/kinect2_calibration#calibrating-the-kinect-one)

## Requirements for live gaze estimation (webcam / RGB only)
- Calibrate your camera (http://wiki.ros.org/camera_calibration). 
- Save the resulting `*.yaml` file to `$(rospack find rt_gene)/webcam_configs/`.
- Change the entry for the `camera_info_url` in the `$(rospack find rt_gene)/launch/start_webcam.launch` file.

# Instructions for estimating gaze

## Estimate gaze live from Kinect One
1) `roscore`
1) `roslaunch rt_gene start_kinect.launch`
1) `roslaunch rt_gene estimate_gaze.launch`

## Estimate gaze live from Webcam / RGB only camera
1) `roscore`
1) `roslaunch rt_gene start_webcam.launch`
1) `roslaunch rt_gene estimate_gaze.launch`

# List of libraries
- `rt_gene/src/rt_gene/detect_face.py`: [MIT License](https://opensource.org/licenses/MIT), [Link to GitHub](https://github.com/davidsandberg/facenet)
- `rt_gene/src/rt_gene/extract_landmarks_new.py` & `rt_gene/src/rt_gene/kalman_stabilizer.py` & `rt_gene/model_nets/frozen_inference_graph.pb`: [MIT License](https://opensource.org/licenses/MIT); [Link to GitHub](https://github.com/yinguobing/head-pose-estimation)
- ROS; [BSD 3-clause](https://opensource.org/licenses/BSD-3-Clause), [Link to website](http://ros.org/)
- Tensorflow; [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), [Link to website](http://tensorflow.org/)
- Keras; [MIT License](https://opensource.org/licenses/MIT), [Link to website](https://keras.io)

