#!/usr/bin/env bash
# echo "Running inference on" ${1}
# echo "Saving Results :" ${2}
# --snapshot pretrained_models/rs19_4000_epoch_274_mean-iu_0.57821.pth

python3 eval.py \
    --dataset railsem19 \
    --arch network.sfnet_resnet.DeepR18_SF_deeply \
    --inference_mode whole \
    --scales 1 \
    --split test \
    --ckpt_path logs/rs19_4000_pretrained_cityscapes_1/test_1 \
    --snapshot pretrained_models/best_epoch_1465_mean-iu_0.75616.pth