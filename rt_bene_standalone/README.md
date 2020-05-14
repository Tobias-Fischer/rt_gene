# RT-BENE: A Dataset and Baselines for Real-Time Blink Estimation in Natural Environments
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
Please follow the steps given in the Requirements section for the [RT-GENE standalone version](../rt_gene_standalone/README.md).

## Basic usage
- Run `$HOME/rt_gene/rt_gene_standalone/estimate_blink_standalone.py`. For supported arguments, run `$HOME/rt_gene_standalone/scripts/estimate_blink_standalone.py --help`

### Optional ensemble model files
- To use an ensemble scheme using multiple models, simply use the `--model` argument, e.g `cd $HOME/rt_gene/ && ./rt_gene_standalone/estimate_blink_standalone.py --models './rt_gene/model_nets/blink_model_1.h5' './rt_gene/model_nets/blink_model_2.h5'`

## List of libraries
See [main README.md](../rt_gene/README.md)

