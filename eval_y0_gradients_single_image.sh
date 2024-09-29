#!/bin/bash

# Set models folder to nfs
export TORCH_HOME="/data/vision/torralba/naturally_robust_models/adversarial_training_imagenet/models"
echo "# TORCH_HOME="$TORCH_HOME
echo ""

# Save datetime as environ variable
# Save datetime as variable
hydra_now=$(date +"%Y-%m-%d_%H-%M-%S")
export HYDRA_NOW=$hydra_now
echo "# HYDRA_NOW="$HYDRA_NOW
echo ""

# Launch torchrun
## Swin-B (Liu2023) Normal
# torchrun --nproc_per_node=1 eval_y0_loss.py --model_name swinb_normal --output-dir outputs_test/intermediate_gradients/swinb_normal --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/val
# Swin-B (Liu2023) Pretrain
# torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode train --model_name swinb_21k  --output-dir outputs_test/eval_y0_gradients_single_image/swinb_21k --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/val
# torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode train --model_name swinb_21k  --output-dir outputs_test/eval_y0_gradients_single_image/swinb_21k --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/train
# Swin-B Liu2023 AT
# torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode train --model_name swinb_at  --output-dir outputs_test/eval_y0_gradients_single_image/swinb_at --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/val
# torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode train --model_name swinb_at  --output-dir outputs_test/eval_y0_gradients_single_image/swinb_at --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/train
# ce w=0.3, Epr(q=0.5) w=-0.1, gradnorm w=0.1
torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode eval --model_name swinb_normal --ckpt-path /data/vision/torralba/naturally_robust_models/adversarial_training_imagenet/outputs/swin_base_patch4_window7_224/2023-11-28_10-04-36/checkpoint-1.pth.tar --output-dir outputs_test/eval_y0_gradients_single_image/epr_median --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/val
torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode eval --model_name swinb_normal --ckpt-path /data/vision/torralba/naturally_robust_models/adversarial_training_imagenet/outputs/swin_base_patch4_window7_224/2023-11-28_10-04-36/checkpoint-1.pth.tar --output-dir outputs_test/eval_y0_gradients_single_image/epr_median --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/train
# ce w=0.3, Epr(q=0.5) w=-0.1, gradnorm w=0.1 + Augmentations
torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode eval --model_name swinb_normal --ckpt-path /data/vision/torralba/naturally_robust_models/adversarial_training_imagenet/outputs/swin_base_patch4_window7_224/2023-11-28_22-12-31/checkpoint-5.pth.tar --output-dir outputs_test/eval_y0_gradients_single_image/epr_median_autoaug --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/val
torchrun --nproc_per_node=1 eval_y0_gradients_single_image.py --mode eval --model_name swinb_normal --ckpt-path /data/vision/torralba/naturally_robust_models/adversarial_training_imagenet/outputs/swin_base_patch4_window7_224/2023-11-28_22-12-31/checkpoint-5.pth.tar --output-dir outputs_test/eval_y0_gradients_single_image/epr_median_autoaug --imagenet_val_path /data/vision/torralba/datasets/imagenet_pytorch/ImageNet/train