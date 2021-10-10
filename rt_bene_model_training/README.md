# RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![stars](https://img.shields.io/github/stars/Tobias-Fischer/rt_gene.svg?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Tobias-Fischer/rt_gene.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/Tobias-Fischer/rt_gene.svg?style=flat-square)

![Best Poster Award](../assets/rt_bene_best_poster_award.png)


## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted; please contact <info@tobiasfischer.info> or <y.demiris@imperial.ac.uk> regarding commercial licensing. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Cortacero_RT-BENE_A_Dataset_and_Baselines_for_Real-Time_Blink_Estimation_in_ICCVW_2019_paper.html):

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
### Tensorflow
For pip users: `pip install tensorflow-gpu numpy tqdm opencv-python scikit-learn` or for conda users: `conda install tensorflow-gpu numpy tqdm opencv scikit-learn`
### Pytorch
For conda users: conda install -c conda-forge numpy scipy tqdm pillow rospkg opencv scikit-learn h5py matplotlib pytorch-lightning && conda install -c pytorch pytorch torchvision

## Model training code
### Tensorflow (as per paper)
This code was used to train the blink estimator for RT-BENE. The labels for the RT-BENE blink dataset are contained in the [rt_bene_dataset](../rt_bene_dataset) directory. The images corresponding to the labels can be downloaded from the RT-GENE dataset (labels are only available for the "noglasses" part): [download](https://zenodo.org/record/2529036) [(alternative link)](https://goo.gl/tfUaDm). Please run `python train_blink_model.py --help` to see the required arguments to train the model.
### Pytorch (experimental)
This code attempts to duplicate the Tensorflow version using Pytorch and Pytorch-Lightning. It uses the same dataset. An HDF5 file is required, to generate this run [GenerateRTBENEH5Dataset.py](pytorch/util/GenerateRTBENEH5Dataset.py) with the argument to locate the RT-BENE dataset.
[train_model](pytorch/train_model.py) contains the code required to train the model in pytorch.
### Eyepatch generation
See [https://github.com/Tobias-Fischer/rt_gene/blob/master/rt_gene_model_training/pytorch/utils/GenerateEyePatchesRTGENEDataset.py](this script) to generate eye patches for model training (only needed on custom datasets)


## Model testing code
### Tensorflow
Evaluation code for a 3-fold evaluation is provided in the [evaluate_blink_model.py](tensorflow/evaluate_blink_model.py) file. An example to train and evaluate an ensemble of models can be found in [train_and_evaluate.py](tensorflow/train_and_evaluate.py). Please run `python train_and_evaluate.py --help` to see the required arguments.

![Results](../assets/rt_bene_precision_recall.png)
### Pytorch
Evaluation code for 3-fold validation is in [evaluate_model.py](pytorch/evaluate_model.py)
![Results](../assets/rtbene_pytorch_resnet18.png)
