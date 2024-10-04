import torch
import torch.nn as nn
from robustbench import benchmark
import torchvision.transforms as transforms
from timm import create_model
import swin_transformer_timm_version

from benchmark_utils import get_Rodriguezmunoz2024Characterizing_model, get_data_transform

transform = get_data_transform()
model = get_Rodriguezmunoz2024Characterizing_model()

threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
dataset = "imagenet"  # one of {"cifar10", "cifar100", "imagenet"}

model_name = "Rodriguezmunoz2024Characterizing"
device = torch.device("cuda:0")

clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                  threat_model=threat_model, eps=8/255, device=device,
                                  preprocessing=transform, data_dir='/data/vision/torralba/datasets/imagenet_pytorch/train',
                                  to_disk=True)