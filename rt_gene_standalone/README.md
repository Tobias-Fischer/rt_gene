# RT-GENE: Real-Time Eye Gaze Estimation in Natural Environments
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg?style=flat-square)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
![HitCount](http://hits.dwyl.io/Tobias-Fischer/rt_gene.svg)
![stars](https://img.shields.io/github/stars/Tobias-Fischer/rt_gene.svg?style=flat-square)
![GitHub issues](https://img.shields.io/github/issues/Tobias-Fischer/rt_gene.svg?style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/Tobias-Fischer/rt_gene.svg?style=flat-square)

## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://openaccess.thecvf.com/content_ECCV_2018/html/Tobias_Fischer_RT-GENE_Real-Time_Eye_ECCV_2018_paper.html):

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

This work was supported in part by the Samsung Global Research Outreach program, and in part by the EU Horizon 2020 Project PAL (643783-RIA).

More information can be found on the Personal Robotic Lab's website: <https://www.imperial.ac.uk/personal-robotics/software/>.

## Requirements
1. Install required Python packages:
    - For `conda` users (recommended): `conda install tensorflow-gpu numpy scipy tqdm pillow opencv && conda install -c pytorch pytorch torchvision`
    - For `pip` users: `pip install tensorflow-gpu numpy scipy tqdm torch torchvision Pillow opencv-python`
1. Download RT-GENE, the required model files and add the source folder to your `PYTHONPATH` environment variable:
    1. `cd $HOME/ && git clone https://github.com/Tobias-Fischer/rt_gene.git`
    1. `python $HOME/rt_gene/rt_gene/scripts/download_models.py`
    1. `export PYTHONPATH=$HOME/rt_gene/rt_gene/src`

## Basic usage
- Run `$HOME/rt_gene/rt_gene/scripts/estimate_gaze_standalone.py`. For supported arguments, run `$HOME/rt_gene/rt_gene/scripts/estimate_gaze_standalone.py --help`

### Optional ensemble model files
- To use an ensemble scheme using 4 models trained on the MPII, UTMV and RT-GENE datasets, simply use the `--models` argument, e.g `cd $HOME/rt_gene/ && ./rt_gene/scripts/estimate_gaze_standalone.py --models './rt_gene/model_nets/all_subjects_mpii_prl_utmv_0_02.h5' './rt_gene/model_nets/all_subjects_mpii_prl_utmv_1_02.h5' './rt_gene/model_nets/all_subjects_mpii_prl_utmv_2_02.h5' './rt_gene/model_nets/all_subjects_mpii_prl_utmv_3_02.h5'`

## List of libraries
See [main README.md](../rt_gene/README.md)

# RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments
## License + Attribution
This code is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Commercial usage is not permitted. If you use this dataset or the code in a scientific publication, please cite the following [paper](http://openaccess.thecvf.com/content_ICCVW_2019/html/GAZE/Cortacero_RT-BENE_A_Dataset_and_Baselines_for_Real-Time_Blink_Estimation_in_ICCVW_2019_paper.html):

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
Please follow the steps given for RT-GENE above.

## Basic usage
- Run `$HOME/rt_gene/rt_gene/scripts/estimate_blink_standalone.py`. For supported arguments, run `$HOME/rt_gene/rt_gene/scripts/estimate_blink_standalone.py --help`

### Optional ensemble model files
- To use an ensemble scheme using multiple models, simply use the `--model` argument, e.g `cd $HOME/rt_gene/ && ./rt_gene/scripts/estimate_blink_standalone.py --models './rt_gene/model_nets/blink_model_1.h5' './rt_gene/model_nets/blink_model_2.h5'`

## List of libraries
See [main README.md](../rt_gene/README.md)

