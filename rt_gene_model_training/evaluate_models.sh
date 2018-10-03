#!/bin/sh

for epoch in 01 02 03 04
do
    # format is: FC1size FC2size FC3size model_type epoch_num GPU_num
    python evaluate_model.py 1024 512 256 512 VGG16 ${epoch} 0
done

