#!/usr/bin/env bash
# echo "Running inference on" ${1}
# echo "Saving Results :" ${2}
python3 demo_floder.py \
    --arch network.sfnet_resnet.DeepR18_SF_deeply \
    --save_dir logs/railsem19_pretrained_cityscapes_0/stylegan2-ada-pytorch_veg/9\
    --snapshot pretrained_models/railsem19_cityscapes_mapillary_epoch_252_mean-iou_0.68545.pth \
    --demo_folder input_data/stylegan2-ada-pytorch/veg/9