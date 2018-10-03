# Requirements
## General
1) Download, install, and configure ROS (full installation; we recommend the Kinectic distribution of ROS, if you want to use a newer version please substitute `kinetic` in all following commands with the correct version name): http://wiki.ros.org/kinetic/Installation

1) `sudo apt-get install python-catkin-tools ros-kinetic-ros-numpy ros-kinetic-camera-info-manager-py ros-kinetic-uvc-camera`

1) `pip install tensorflow-gpu keras numpy opencv-python scipy tqdm`

1) `cd $HOME/catkin_ws/src && git clone https://github.com/Tobias-Fischer/rt_gene.git`

1) `cd $HOME/catkin_ws && catkin build`

1) To use a single model trained on the RT-GENE dataset, [download model file (two eyes)](https://imperialcollegelondon.box.com/s/zu424pzptmw1klh70jsc697b37h7mwif) and save the file in `$HOME/catkin_ws/src/rt_gene/rt_gene/model_nets/`.

1) To use an ensemble scheme using 4 models trained on the MPII, UTMV and RT-GENE datasets, download the following files (make sure you comply with the licenses of [MPII](https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/) and [UTMV](http://www.hci.iis.u-tokyo.ac.jp/datasets/)!): [model 1](https://imperialcollegelondon.box.com/s/5cjnijpo8qxawbkik0gjrmyc802j2h1v), [model 2](https://imperialcollegelondon.box.com/s/1ye5jlh5ce11f93yn1s36uysjta7a3ob), [model 3](https://imperialcollegelondon.box.com/s/5vl9samndju9zhygtai8z6kkpw2jmjll), [model 4](https://imperialcollegelondon.box.com/s/hmcoxopu4xetic5bm47xqrl5mqktpg92). Then, save the files in `$HOME/catkin_ws/src/rt_gene/rt_gene/model_nets/`. Finally, open `$HOME/catkin_ws/src/rt_gene/rt_gene/launch/estimate_gaze.launch` and change `<rosparam param="model_files">['model_nets/Model_allsubjects1.h5']</rosparam>` to `<!--rosparam param="model_files">['model_nets/Model_allsubjects1.h5']</rosparam-->` and `<!--rosparam param="model_files">['model_nets/all_subjects_mpii_prl_utmv_0_02.h5', 'model_nets/all_subjects_mpii_prl_utmv_1_02.h5', 'model_nets/all_subjects_mpii_prl_utmv_2_02.h5', 'model_nets/all_subjects_mpii_prl_utmv_3_02.h5']</rosparam-->` to `<rosparam param="model_files">['model_nets/all_subjects_mpii_prl_utmv_0_02.h5', 'model_nets/all_subjects_mpii_prl_utmv_1_02.h5', 'model_nets/all_subjects_mpii_prl_utmv_2_02.h5', 'model_nets/all_subjects_mpii_prl_utmv_3_02.h5']</rosparam>`

1) (optional) [Download head pose caffemodel](https://github.com/yinguobing/head-pose-estimation/raw/master/assets/res10_300x300_ssd_iter_140000.caffemodel) and save the file in `$HOME/catkin_ws/src/rt_gene/rt_gene/model_nets/`

1) (optional) [Download head pose inference graph](https://github.com/yinguobing/head-pose-estimation/raw/master/assets/frozen_inference_graph.pb) and save the file in `$HOME/catkin_ws/src/rt_gene/rt_gene/model_nets/`

## Requirements for live gaze estimation (Kinect One)
- Follow instructions for https://github.com/code-iai/iai_kinect2
- Make sure the calibration is saved correctly (https://github.com/code-iai/iai_kinect2/tree/master/kinect2_calibration#calibrating-the-kinect-one)

## Requirements for live gaze estimation (webcam / RGB only)
- Calibrate your camera (http://wiki.ros.org/camera_calibration). Then, save the resulting `*.yaml` file to `$HOME/catkin_ws/src/rt_gene/rt_gene/` and change the entry for the `camera_info_url` in the `$HOME/catkin_ws/src/rt_gene/rt_gene/launch/start_webcam.launch` file.

# Instructions for estimating gaze

## Estimate gaze live from Kinect One
1) Terminal 1 - `roscore`

1) Terminal 2 - `roslaunch rt_gene start_kinect.launch`

1) Terminal 3 - `roslaunch rt_gene estimate_gaze.launch`

## Estimate gaze live from Webcam / RGB only camera
1) Terminal 1 - `roscore`

1) Terminal 2 - `roslaunch rt_gene start_webcam.launch`

1) Terminal 3 - `roslaunch rt_gene estimate_gaze.launch`

# List of libraries
- `rt_gene_inpainting/external/poissonblending.py`; [MIT License](https://opensource.org/licenses/MIT); [Link to GitHub](https://github.com/parosky/poissonblending)
- `rt_gene/src/rt_gene/detect_face.py` [MIT License](https://opensource.org/licenses/MIT), [Link to GitHub](https://github.com/davidsandberg/facenet)
- `rt_gene/src/rt_gene/extract_landmarks_new.py` & `rt_gene/src/rt_gene/kalman_stabilizer.py`; [MIT License](https://opensource.org/licenses/MIT); [Link to GitHub](https://github.com/yinguobing/head-pose-estimation)
- ROS; [BSD 3-clause](https://opensource.org/licenses/BSD-3-Clause), [Link to website](http://ros.org/)
- Tensorflow; [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), [Link to website](http://tensorflow.org/)
- Keras; [MIT License](https://opensource.org/licenses/MIT), [Link to website](https://keras.io)
