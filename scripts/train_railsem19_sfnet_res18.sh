# CUDA_VISIBLE_DEVICES="4,5,6,7" sh ./scripts/train_railsem19_sfnet_res18.sh

#!/usr/bin/env bash
now=$(date +"%Y%m%d_%H%M%S")
EXP_DIR=./logs/rs19_trainVal_pretrained_cityscapes_3
mkdir -p ${EXP_DIR}
# Example on Cityscapes by resnet50-deeplabv3+ as baseline
python3 -m torch.distributed.launch --nproc_per_node=4 train.py \
  --dataset railsem19 \
  --cv 0 \
  --arch network.sfnet_resnet.DeepR18_SF_deeply \
  --class_uniform_pct 0.5 \
  --class_uniform_tile 1080 \
  --lr 0.0025 \
  --lr_schedule poly \
  --poly_exp 1.0 \
  --repoly 1.5  \
  --rescale 1.0 \
  --syncbn \
  --sgd \
  --ohem \
  --crop_size 1080 \
  --scale_min 0.5 \
  --scale_max 2.0 \
  --color_aug 0.25 \
  --gblur \
  --bblur \
  --max_epoch 400 \
  --wt_bound 1.0 \
  --bs_mult 8 \
  --apex \
  --exp railsem19_SFnet_res18_lr_0.002 \
  --ckpt ${EXP_DIR}/ \
  --tb_path ${EXP_DIR}/ \
  --snapshot pretrained_models/pretrained_cityscapes_mapillary_rs18_miou-0.799.pth \
  2>&1 | tee  ${EXP_DIR}/log_${now}.txt &
