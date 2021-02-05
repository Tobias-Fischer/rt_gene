#!/usr/bin/env bash

/home/ahmed/miniconda3/envs/venv3/bin/python /home/ahmed/catkin_ws/src/rt_gene/rt_gene_model_training/pytorch/train_model.py --hdf5_file="/home/ahmed/combined_dataset.hdf5" --dataset="other" --batch_size=32 --accumulate_grad_batches=1 --model_base="vgg" --num_io_workers=6 --learning_rate=0.000325