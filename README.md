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

# Overview
The code is split into three parts, each having its own README contained. There is also an accompanying dataset to the code.

## rt_gene (ROS package)
The `rt_gene` directory contains a ROS package for real-time eye gaze estimation. This contains all the code required at inference time.

## rt_gene_inpainting
The `rt_gene_inpainting` directory contains code to inpaint the region covered by the eyetracking glasses.

## rt_gene_model_training
The `rt_gene_model_training` directory allows using the inpainted images to train a deep neural network for eye gaze estimation.

