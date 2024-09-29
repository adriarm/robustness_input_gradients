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
# torchrun --nproc_per_node=2 --master_port $MASTER_PORT adversarial_training_cifar.py --configs=./arxiv_configs_cifar/nattrain_prn18_gelu.yaml
# torchrun --nproc_per_node=2 --master_port $MASTER_PORT adversarial_training_cifar.py --configs=./arxiv_configs_cifar/regtrain_prn18_gelu.yaml
# torchrun --nproc_per_node=2 --master_port $MASTER_PORT adversarial_training_cifar.py --configs=./arxiv_configs_cifar/advtrain_prn18_gelu.yaml
# torchrun --nproc_per_node=2 --master_port $MASTER_PORT adversarial_training_cifar.py --configs=./arxiv_configs_cifar/gradnorm_prn18_gelu.yaml