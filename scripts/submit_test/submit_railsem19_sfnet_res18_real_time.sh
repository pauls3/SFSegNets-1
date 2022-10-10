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
    --ckpt_path logs/pretrained_cityscapes_1_test_0 \
    --snapshot logs/rs19_trainVal_pretrained_cityscapes_2/railsem19_SFsegnet_res18_lr_0.00015/rail-network.sfnet_resnet.DeepR18_SF_deeply_apex_T_bblur_T_bs_mult_8_class_uniform_tile_1080_crop_size_1080_cv_0_dataset_railsem_lr_0.001_ohem_T_PT_sbn/best_epoch_1465_mean-iu_0.75616.pth