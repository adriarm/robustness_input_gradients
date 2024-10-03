# Characterizing robustness with natural input gradients

This repository contains the code for the paper "Characterizing Robustness via Natural Input Gradients". 

<p align="center">
  <img width="100%" src="https://adriarm.github.io/_pages/robustness_input_gradients/images/zzz_gradient_comparison_poster_white.png">
</p>

[[Project page](https://adriarm.github.io/_pages/robustness_input_gradients/)] 
[[Paper](https://arxiv.org/abs/XXXX.XX)]

# Requirements
## Conda
Set up the conda environment as follows:
```
conda create -n RIG python=3.9 -y
conda activate RIG

conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install time pyyaml scipy gdown
```

## Pip
ToDo

# Download models
For the models referenced in the paper, they can be downloaded from the following links
| Model    | Clean acc. | AutoAttack (standard, $\epsilon=\frac{4}{255}$) acc. |
| -------- | ------- | ------- |
| [GradNorm - SwinB](https://drive.google.com/file/d/1OHHY9ulzWGkCJdhvCM2UmDv3OvEKCzp2/view?usp=sharing)  |  77.78    | 51.58 |
| [EdgeReg - SwinB](https://drive.google.com/file/d/1gYwtIt6I6b-CKO-TvNQ3qbOFmN2B7Izw/view?usp=sharing) | 76.80     | 35.02|
| [GradNorm - ResNet50+GeLU](https://drive.google.com/file/d/1CvLhHaFVyqmqL6W0P_-H8iw2R_uk8MbM/view?usp=share_link) | 60.34 | 30.00|

# Data
The ImageNet dataset is needed, which can be downloaded from https://www.image-net.org

# Training
The main training code borrows the vast majority of the content from [ARES-Bench](https://github.com/thu-ml/adversarial_training_imagenet) with minor code and training recipe modifications. Our models can be reproduced by running `run_train.sh`. Note that the `train_dir` and `eval_dir` ImageNet locations in the config files (./configs_cifar, ./configs_finetuning, ./configs_liu2023, ./configs_logit, ./configs_train) will need to be changed to yours. 

# Evaluation
The evaluation is exactly the same as [ARES-Bench](https://github.com/thu-ml/adversarial_training_imagenet) for consistency. Our results can be reproduced by running `run_eval.sh`. Simply replace YOUR_MODEL_PATH and YOUR_IMAGENET_VAL_PATH for your own values.

# Citation
```
@inproceedings{rodriguezmunoz2024characterizing,
  title={Characterizing model robustness via natural input gradients},
  author={Adrián Rodríguez-Muñoz and Tongzhou Wang and Antonio Torralba},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024},
  url={}
}
```
