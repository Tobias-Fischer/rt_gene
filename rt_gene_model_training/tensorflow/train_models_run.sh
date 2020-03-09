#!/bin/sh

for ensemble_num in 1 2 3 4
do
    # format is: FC1size FC2size FC3size model_type ensemble_num GPU_num
    python train_model.py 1024 512 256 512 VGG16 ${ensemble_num} 0
end

