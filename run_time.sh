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
## Debug
# torchrun --nproc_per_node=2 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/debug.yaml

## SwinB
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/nattrain_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/gradnorm_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain1_swinb_time.yaml

hydra_now=$(date +"%Y-%m-%d_%H-%M-%S")
export HYDRA_NOW=$hydra_now
echo "# HYDRA_NOW="$HYDRA_NOW
echo ""

torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/gradnorm_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain1_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/nattrain_swinb_time.yaml

hydra_now=$(date +"%Y-%m-%d_%H-%M-%S")
export HYDRA_NOW=$hydra_now
echo "# HYDRA_NOW="$HYDRA_NOW
echo ""

torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain1_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/nattrain_swinb_time.yaml
torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/gradnorm_swinb_time.yaml

## SwinS
# torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/gradnorm_swins.yaml
# torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./train_configs/advtrain_swins.yaml

