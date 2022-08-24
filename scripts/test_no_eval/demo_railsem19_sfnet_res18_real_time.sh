#!/usr/bin/env bash
# echo "Running inference on" ${1}
# echo "Saving Results :" ${2}
# --snapshot pretrained_models/railsem19_cityscapes_mapillary_epoch_252_mean-iou_0.68545.pth \
# --demo_folder /home/stanik/rtis_lab/data/RailSem19/custom_split/test_images
python3 demo_floder.py \
    --arch network.sfnet_resnet.DeepR18_SF_deeply \
    --save_dir logs/railsem19_pretrained_cityscapes_0/no_eval_test_2 \
    --snapshot logs/rs19_trainVal_pretrained_cityscapes_2/test_model.pth \
    --demo_folder input_data/stylegan2-ada-pytorch/test