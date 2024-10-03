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

# Swin-B
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/gradnorm_swinb_variant_test.yaml
# torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/advtrain_swinb.yaml
torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/gradnorm_swinb_variant.yaml
# torchrun --nproc_per_node=8 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/advtrain1_swinb.yaml
# torchrun --nproc_per_node=8 adversarial_training_logit.py --configs=./configs_logit/logitsobel_swinb_hard05.yaml

# ResNet-50
## Finetune non-linearity switch
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/finetune_resnet_gelu.yaml
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/finetune_resnet_relu.yaml
# torchrun --nproc_per_node=2 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/finetune_resnet_silu.yaml
## AdvTrain
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/advtrain_resnet_gelu.yaml
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/advtrain_resnet_relu.yaml
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/advtrain_resnet_silu.yaml
## GradNorm
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/gradnorm_resnet_gelu.yaml
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/gradnorm_resnet_relu.yaml
# torchrun --nproc_per_node=4 --master_port $MASTER_PORT adversarial_training.py --configs=./configs_train/gradnorm_resnet_silu.yaml
## EdgeReg
# torchrun --nproc_per_node=7 adversarial_training_logit.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05.yaml
# torchrun --nproc_per_node=7 adversarial_training_logit.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05_saliency.yaml
# torchrun --nproc_per_node=7 adversarial_training_logit_loss.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05_loss.yaml