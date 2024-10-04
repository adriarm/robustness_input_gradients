import torch
import torch.nn as nn
from robustbench import benchmark
import torchvision.transforms as transforms
from timm import create_model

import swin_transformer_timm_version

def get_data_transform():
  # Transform args
  input_size = 224
  crop_pct = 0.875
  interpolation = 3

  t = []
  if input_size > 32:
      size = int(input_size/crop_pct)
      t.append(
          transforms.Resize(size, interpolation=interpolation),
      )
      t.append(transforms.CenterCrop(input_size))
  else:
      t.append(
          transforms.Resize(input_size, interpolation=interpolation),
      )
  t.append(transforms.ToTensor())
  test_transform = transforms.Compose(t)
  return test_transform

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def get_RodriguezMunoz2024Characterizing_model(ckpt_location):
  model = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=1000, act_gelu=True)
  ckpt=torch.load(ckpt_location, map_location='cpu')
  model.load_state_dict(ckpt['state_dict_ema'])

  normalize = NormalizeByChannelMeanStd(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
  model = torch.nn.Sequential(normalize, model)
  return model.eval()