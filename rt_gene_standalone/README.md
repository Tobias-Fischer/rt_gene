# License + Attribution
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

# Requirements
## Manual installation
1. Install required Python packages:
    - For `conda` users (recommended): `conda install tensorflow-gpu keras numpy scipy tqdm pytorch torchvision pillow && conda install -c 1adrianb face_alignment`
    - For `pip` users: `pip install tensorflow-gpu keras numpy scipy tqdm torch torchvision Pillow face-alignment`
1. Download RT-GENE, the required model files and add the source folder to your `PYTHONPATH` environment variable:
    1. `cd $HOME/ && git clone https://github.com/Tobias-Fischer/rt_gene.git`
    1. `python $HOME/rt_gene/rt_gene/scripts/download_models.py`
    1. `export PYTHONPATH=$HOME/rt_gene/rt_gene/src`

## Optional ensemble model files
- To use an ensemble scheme using 4 models trained on the MPII, UTMV and RT-GENE datasets, please follow instructions [here](../rt_gene/README.md#optional-ensemble-model-files)

# Instructions for estimating gaze
- Run `$HOME/rt_gene/rt_gene/scripts/estimate_gaze_standalone.py`. For supported arguments, run `$HOME/rt_gene/rt_gene/scripts/estimate_gaze_standalone.py --help`

# List of libraries
See [main README.md](../rt_gene/README.md)

