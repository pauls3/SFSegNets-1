#!/usr/bin/env bash
# echo "Running inference on" ${1}
# echo "Saving Results :" ${2}
python3 demo_floder.py \
    --arch network.sfnet_resnet.DeepR18_SF_deeply \
    --save_dir logs/railsem19_pretrained_cityscapes_0/no_eval_test_nevada_0 \
    --snapshot pretrained_models/railsem19_cityscapes_mapillary_epoch_252_mean-iou_0.68545.pth \
    --demo_folder /home/stanik/datasets/nevada_test_images