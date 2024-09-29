#!/bin/bash

# Set models folder to nfs
# export TORCH_HOME="/vision-nfs/torralba/projects/adrianr/input_norm/models"
echo "# TORCH_HOME="$TORCH_HOME
echo ""

# Save datetime as environ variable
# Save datetime as variable
hydra_now=$(date +"%Y-%m-%d_%H-%M-%S")
export HYDRA_NOW=$hydra_now
echo "# HYDRA_NOW="$HYDRA_NOW
echo ""

export MASTER_PORT=$(( 10000 + ($$ % 50000) ))
echo "# MASTER_PORT="$MASTER_PORT

# Launch torchrun
torchrun --nproc_per_node=8 --master_port $MASTER_PORT eval_white_box_cifar.py --model_name prn18 --ckpt-path YOUR_MODEL_PATH --attack_types autoattack --batch_size 1024 --imagenet_val_path YOUR_IMAGENET_VAL_PATH

