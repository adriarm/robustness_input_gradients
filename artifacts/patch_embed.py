# Add non-linearity and anti-aliased down-sampling
""" Image to Patch Embedding using Conv2d

A convolution based approach to patchifying a 2D image w/ embedding projection.

Based on the impl in https://github.com/google-research/vision_transformer

Hacked together by / Copyright 2020 Ross Wightman
"""
import math

import numpy as np
import scipy
import torch
from timm.models.layers.helpers import to_2tuple
from timm.models.layers.trace_utils import _assert
from torch import nn as nn


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=1)
        self.act = nn.GELU()
        self.register_buffer('blur_kernel', torch.Tensor([0.25, 0.5, 0.25]), persistent=False)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        _assert(H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]}).")
        _assert(W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]}).")
        # Project
        x = self.proj(x)
        # GELU
        # x = self.act(x)
        # Blur + Down
        x = filter_separable_2d_with_reflect_pad(x, self.blur_kernel)
        x = x[:, :, ::self.patch_size[0], ::self.patch_size[1]]
        # Reshape
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def filter_separable_2d_with_reflect_pad(x, h):
  x = filter_2d_with_reflect_pad(x, h[None, :])
  x = filter_2d_with_reflect_pad(x, h[:, None])
  return x

def filter_2d_with_reflect_pad(x, h):
  _, C, _, _ = x.shape
  p2d = tuple([(h.size(1)-1)//2] * 2 + [(h.size(0)-1)//2] * 2)
  x = nn.functional.pad(x, p2d, "reflect")
  return nn.functional.conv2d(x, h[None, None, :, :].expand((C, -1, -1, -1)), padding='valid', groups=C)

def make_gaussian_filter(stddev):
  order = math.ceil(3*stddev)
  n = np.arange(-order, order+1)
  h = math.exp(-stddev) * scipy.special.iv(n, stddev)
  h = np.array(h, dtype=np.float32)
  h = h / np.sum(h)
  return h
