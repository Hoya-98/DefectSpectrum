#!/bin/sh
CUDA_VISIBLE_DEVICES='0' \
python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=29511 \
inference.py \
--step_inference 400 \
--sample_dir './defect_pills/' \
--large_recep './defect_pills_large/checkpoint/diffusion_010000.pt' \
--small_recep './defect_pills_small/checkpoint/diffusion_020000.pt' \
--num_defect 1 \
--large_recep_config './config/large_recep.yml' \
--small_recep_config './config/small_recep.yml' \
