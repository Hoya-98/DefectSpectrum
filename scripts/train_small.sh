CUDA_VISIBLE_DEVICES="0" \
nohup python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_port=29516 \
  train.py \
  --config ./config/small_recep.yml \
  --work_dir ./pills_dataset \
  --seperate_channel_loss 0 \
  --num_defect 1 \
> train.log 2>&1 &

