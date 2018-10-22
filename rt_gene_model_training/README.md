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
`pip install tensorflow-gpu keras numpy scipy tqdm matplotlib h5py scikit-learn`

# Model training code
This code was used to train the eye gaze estimation CNN for RT-GENE. 
- First, the h5 files need to be created from the RAW images. We use the `prepare_dataset.m` MATLAB script for this purpose. Please adjust the `load_path` and `save_path` variables. The `augmented` variable can be set to `0` to disable image image augmentations described in the paper. The `with_faces` variable can be set to `1` to also store the face images in the *.h5 files (warning: this requires a lot of memory).
- Then, the `train_model.py` file can be used to train the models in the 3-Fold setting as described in the paper. An example to call this script is given in the `train_models_run.sh` file.
- Finally, the `evaluate_model.py` can be used to get the individual models' performance as well as the ensemble performance. An example to call this script is given in the `evaluate_models.sh` file.

# List of libraries
- Tensorflow; [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0), [Link to website](http://tensorflow.org/)
- Keras; [MIT License](https://opensource.org/licenses/MIT), [Link to website](https://keras.io)

