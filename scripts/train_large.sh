#!/bin/sh
CUDA_VISIBLE_DEVICES="0" \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=29513 \
train.py \
--work_dir ./defect_pills/ \
--config ./config/large_recep.yml \
--seperate_channel_loss 0 \
--num_defect 1
