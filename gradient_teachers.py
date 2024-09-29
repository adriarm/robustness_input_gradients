import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F


class ContourEnergy(nn.Module):
  def __init__(self, sigma, k_sobel) -> None:
    super().__init__()

    # Gaussian kernel
    gaussian_kernel = make_gaussian_filter(sigma)
    self.register_buffer('gaussian_kernel', torch.from_numpy(gaussian_kernel))

    # Sobel kernel
    sobel_kernel = make_sobel_kernel(k_sobel)
    self.register_buffer('sobel_kernel', torch.from_numpy(sobel_kernel))

  def forward(self, img, dz=None):
    N, C, H, W = img.shape
    
    # Blur
    p2d = tuple([(self.gaussian_kernel.size(0)-1)//2] * 4)
    img = F.pad(img, p2d, "reflect")
    img = F.conv2d(img, self.gaussian_kernel[None, None, :, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    img = F.conv2d(img, self.gaussian_kernel[None, None, None, :].expand((C, -1, -1, -1)), padding='valid', groups=C)
    if dz is not None:
      dz = F.pad(dz, p2d, "reflect")
      dz = F.conv2d(dz, self.gaussian_kernel[None, None, :, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
      dz = F.conv2d(dz, self.gaussian_kernel[None, None, None, :].expand((C, -1, -1, -1)), padding='valid', groups=C)
    # img = F.conv2d(img, self.gaussian_kernel[None, None, :, None].expand((C, -1, -1, -1)), padding='same', groups=C)
    # img = F.conv2d(img, self.gaussian_kernel[None, None, None, :].expand((C, -1, -1, -1)), padding='same', groups=C)
    
    # Sobel
    p2d = tuple([(self.sobel_kernel.size(0)-1)//2] * 4)
    img = F.pad(img, p2d, "reflect")
    dx = F.conv2d(img, self.sobel_kernel[None, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    dy = F.conv2d(img, self.sobel_kernel.T[None, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    if dz is not None:
      dz = F.pad(dz, p2d, "reflect")
      dzx = F.conv2d(dz, self.sobel_kernel[None, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
      dzy = F.conv2d(dz, self.sobel_kernel.T[None, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    # dx = F.conv2d(img, self.sobel_kernel[None, None].expand((C, -1, -1, -1)), padding='same', groups=C)
    # dy = F.conv2d(img, self.sobel_kernel.T[None, None].expand((C, -1, -1, -1)), padding='same', groups=C)
    
    # Square and sum
    E = (dx**2 + dy**2).sum(-3, keepdim=True)**0.5
    if dz is not None:
      dz = (dx*dzx + dy*dzy)/ (1e-12 + E)

    # Blur (for centering)
    p2d = tuple([(self.gaussian_kernel.size(0)-1)//2] * 4)
    E = F.pad(E, p2d, "reflect")
    E = F.conv2d(E, self.gaussian_kernel[None, None, :, None], padding='valid')
    E = F.conv2d(E, self.gaussian_kernel[None, None, None, :], padding='valid')
    if dz is not None:
      dz = F.pad(dz, p2d, "reflect")
      dz = F.conv2d(dz, self.gaussian_kernel[None, None, :, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
      dz = F.conv2d(dz, self.gaussian_kernel[None, None, None, :].expand((C, -1, -1, -1)), padding='valid', groups=C)


    # Expand to orignal number of channels
    E = E.expand(-1, C, -1, -1)

    if dz is not None:
      return E, dz
    else:
      return E

def make_gaussian_filter(stddev):
  order = math.ceil(3*stddev)
  n = np.arange(-order, order+1)
  h = math.exp(-stddev) * scipy.special.iv(n, stddev)
  h = np.array(h, dtype=np.float32)
  h = h / np.sum(h)
  return h

def make_sobel_kernel(k=3):
    # get range
    range = np.linspace(-(k // 2), k // 2, k)
    # compute a grid the numerator and the axis-distances
    x, y = np.meshgrid(range, range)
    sobel_2D_numerator = x
    sobel_2D_denominator = (x ** 2 + y ** 2)
    sobel_2D_denominator[:, k // 2] = 1  # avoid division by zero
    sobel_2D = sobel_2D_numerator / sobel_2D_denominator
    sobel_2D = np.array(sobel_2D, dtype=np.float32)
    return sobel_2D