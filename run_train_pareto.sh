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

## SwinB
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_control.yaml

# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_green.yaml

# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_randompert_2.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_randompert_4.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_randompert_8.yaml

# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_08.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_10.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_11.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_12.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_13.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_14.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_15.yaml
# torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_16.yaml
#torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_17.yaml
#torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_18.yaml
#torchrun --nproc_per_node=7 --master_port $MASTER_PORT adversarial_training_finetuning.py --configs=./configs_finetuning/gradnorm_swinb_pareto_19.yaml

