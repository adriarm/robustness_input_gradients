import torch

from robustbench import benchmark
from myrobust model import MyRobustModel

threat_model = "Linf"  # one of {"Linf", "L2", "corruptions"}
dataset = "imagenet"  # one of {"cifar10", "cifar100", "imagenet"}

model = MyRobustModel()
model_name = "<Name><Year><FirstWordOfTheTitle>"
device = torch.device("cuda:0")

clean_acc, robust_acc = benchmark(model, model_name=model_name, n_examples=10000, dataset=dataset,
                                  threat_model=threat_model, eps=8/255, device=device,
                                  to_disk=True)