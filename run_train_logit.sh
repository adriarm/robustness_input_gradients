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

# Launch torchrun
# torchrun --nproc_per_node=8 adversarial_training_logit.py --configs=./my_configs/logit_training_gradacum.yaml
# torchrun --nproc_per_node=4 adversarial_training_logit.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05.yaml
# torchrun --nproc_per_node=7 adversarial_training_logit.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05.yaml
# torchrun --nproc_per_node=7 adversarial_training_logit.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05_saliency.yaml
torchrun --nproc_per_node=7 adversarial_training_logit_loss.py --configs=./configs_logit/logitsobel_resnet_gelu_hard05_loss.yaml
# torchrun --nproc_per_node=7 adversarial_training_logit.py --configs=./configs_logit/logitsobel_swinb_hard05.yaml