import math

import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from functorch import vmap

# veigh = vmap(torch.linalg.eigh) 

class DBP(nn.Module):
  def __init__(self, eps=4./255., std=0.225) -> None:
    super().__init__()
    self.eps = eps/std

  def forward(self, gradients, inputs):
    batch_size = gradients.shape[0]
    return self.eps*batch_size*gradients.abs().sum((-3, -2, -1)).mean()

# class DBPAMHM(nn.Module):
#   def __init__(self, eps=4./255., std=0.225, w_hm=0.25e1, tol=1e-12) -> None:
#     super().__init__()
#     self.eps = eps/std
#     self.w_hm = w_hm
#     self.tol = tol

#   def forward(self, gradients, inputs):
#     batch_size = gradients.shape[0]
#     gradients = self.eps*batch_size*gradients.abs().flatten(1)
#     am = gradients.sum(1).mean()
#     hm = self.w_hm * (gradients.clamp(min=self.tol).pow(-1).mean(-1).pow(-1).mean() - self.tol)
#     return am, hm

#   def __repr__(self):
#     return f'{self.__class__.__name__}(w_hm={self.w_hm}, tol={self.tol})'

# class DBPSparsity(nn.Module):
#   def __init__(self, eps=4./255., std=0.225) -> None:
#     super().__init__()
#     self.eps = eps/std

#   def forward(self, gradients, inputs):
#     batch_size = gradients.shape[0]
#     l1 = gradients.abs().sum((-3, -2, -1))
#     l2 = gradients.abs().square().sum((-3, -2, -1)).sqrt()
#     norm_term = self.eps*batch_size*l1.mean()
#     sparsity_term = 0.005* (l1/l2).mean()
#     return norm_term, sparsity_term

class DBPChannel(nn.Module):
  def __init__(self, eps=4./255., std=0.225, weight_r=1., weight_g=2., weight_b=1.) -> None:
    super().__init__()
    self.eps = eps/std
    self.weight_r = weight_r
    self.weight_g = weight_g
    self.weight_b = weight_b

  def forward(self, gradients, inputs):
    batch_size = gradients.shape[0]
    gradients = self.eps*batch_size*gradients.abs().sum((-2, -1)).mean(0)
    return self.weight_r * gradients[0] + self.weight_g * gradients[1] + self.weight_b * gradients[2]

  def __repr__(self):
    return f'{self.__class__.__name__}(weight_r={self.weight_r}, weight_g={self.weight_g}, weight_b={self.weight_b})'

class DBPThresholded(nn.Module):
  def __init__(self, th=0.001, eps=4./255., std=0.225) -> None:
    super().__init__()
    self.th = th
    self.eps = eps/std

  def forward(self, gradients, inputs):
    batch_size = gradients.shape[0]
    return torch.relu(batch_size*self.eps*gradients.abs().sum((-3, -2, -1)) - self.th).mean()

# class DBPPow(nn.Module):
#   def __init__(self, eps=4./255., std=0.225, p=0.7, th=0.01, tol=1e-12) -> None:
#     super().__init__()
#     self.eps = eps/std
#     self.p = p
#     self.th = th
#     self.tol = tol

#   def forward(self, gradients, inputs):
#     batch_size = gradients.shape[0]
#     return torch.relu(self.eps*batch_size*(gradients.abs() + self.tol).pow(self.p).sum((-3, -2, -1)) - self.th).mean()

#   def __repr__(self):
#     return f'{self.__class__.__name__}(p={self.p}, th={self.th}, tol={self.tol})'

# class DBPEdgeWeight(nn.Module):
#   def __init__(self, eps=4./255., std=0.225, theta=0.1, rho=0.5) -> None:
#     super().__init__()
#     self.eps = eps/std
#     self.theta = theta
#     self.rho = rho

#     self.register_buffer('inner_gaussian', torch.from_numpy(make_gaussian_filter(self.theta)))
#     self.register_buffer('left_diff', torch.Tensor([[-1., 1., 0.]]), persistent=False)
#     self.register_buffer('right_diff', torch.Tensor([[0., -1., 1.]]), persistent=False)
#     self.register_buffer('outer_gaussian', torch.from_numpy(make_gaussian_filter(self.rho)))

#   def forward(self, gradients, inputs):
#     batch_size = gradients.shape[0]
#     with torch.no_grad():
#       SI = structure_tensor(inputs, self.inner_gaussian, self.left_diff, self.right_diff, self.outer_gaussian)
#       p_edge = get_edge_probability(SI)
#       q_edge = 1. - p_edge
#     return self.eps*batch_size*(q_edge * gradients).abs().sum((-3, -2, -1)).mean()

# class DBPEdgeWeightNorm(nn.Module):
#   def __init__(self, eps=4./255., std=0.225, theta=0.1, rho=0.5) -> None:
#     super().__init__()
#     self.eps = eps/std
#     self.theta = theta
#     self.rho = rho

#     self.register_buffer('inner_gaussian', torch.from_numpy(make_gaussian_filter(self.theta)))
#     self.register_buffer('left_diff', torch.Tensor([[-1., 1., 0.]]), persistent=False)
#     self.register_buffer('right_diff', torch.Tensor([[0., -1., 1.]]), persistent=False)
#     self.register_buffer('outer_gaussian', torch.from_numpy(make_gaussian_filter(self.rho)))

#   def forward(self, gradients, inputs):
#     batch_size = gradients.shape[0]
#     with torch.no_grad():
#       SI = structure_tensor(inputs, self.inner_gaussian, self.left_diff, self.right_diff, self.outer_gaussian)
#       p_edge = get_edge_probability(SI)
#       q_edge = 1. - p_edge
#       q_edge = q_edge / q_edge.mean((-3, -2, -1), keepdim=True)
#     return self.eps*batch_size*(q_edge * gradients).abs().sum((-3, -2, -1)).mean()

# class DBPTangent(nn.Module):
#   def __init__(self, eps=4./255., std=0.225, theta=0.1, rho=0.5) -> None:
#     super().__init__()
#     self.eps = eps/std
#     self.theta = theta
#     self.rho = rho

#     self.register_buffer('inner_gaussian', torch.from_numpy(make_gaussian_filter(self.theta)))
#     self.register_buffer('left_diff', torch.Tensor([[-1., 1., 0.]]), persistent=False)
#     self.register_buffer('right_diff', torch.Tensor([[0., -1., 1.]]), persistent=False)
#     self.register_buffer('outer_gaussian', torch.from_numpy(make_gaussian_filter(self.rho)))

#   def forward(self, gradients, inputs):
#     SI = structure_tensor(inputs, self.inner_gaussian, self.left_diff, self.right_diff, self.outer_gaussian)
#     #print(SI.shape)
#     tangential = get_tangential_direction(SI)
#     #print(tangential.shape)
#     gradients_dx = filter_2d_with_reflect_pad(gradients, self.left_diff)
#     gradients_dy = filter_2d_with_reflect_pad(gradients, self.left_diff.T)
#     gradients_dtan = tangential[:, None, ..., 0]*gradients_dx + tangential[:, None, ..., 1]*gradients_dy

#     batch_size = gradients.shape[0]
#     d0_term = batch_size*self.eps*gradients.abs().sum((-3, -2, -1)).mean()
#     d1_term = batch_size*self.eps*gradients_dtan.abs().sum((-3, -2, -1)).mean()
#     return d0_term, d1_term

# class DBPChange(nn.Module):
#   def __init__(self, eps=4./255., std=0.225) -> None:
#     super().__init__()
#     self.eps = eps/std

#     self.register_buffer('left_diff', torch.Tensor([[-1., 1., 0.]]), persistent=False)
#     self.register_buffer('right_diff', torch.Tensor([[0., -1., 1.]]), persistent=False)

#   def forward(self, gradients, inputs):
#     batch_size = gradients.shape[0]
#     gradients_dx = filter_2d_with_reflect_pad(gradients, self.left_diff)
#     gradients_dy = filter_2d_with_reflect_pad(gradients, self.left_diff.T)

#     d0_term = self.eps*batch_size*gradients.abs().sum((-3, -2, -1)).mean()
#     d1_term = batch_size*self.eps*0.5*(gradients_dx.abs() + gradients_dy.abs()).sum((-3, -2, -1)).mean()
#     return d0_term, d1_term


# ## Structure tensor
# def get_tangential_direction(SI):
#   with torch.no_grad():
#     eigenvalues, eigenvectors = veigh(SI)
#     uv = eigenvectors[..., 0, :] #.chunk(2, dim=-1)
#   return uv

# def get_orthogonal_direction(SI):
#   with torch.no_grad():
#     eigenvalues, eigenvectors = veigh(SI)
#     uv = eigenvectors[..., 1, :] #.chunk(2, dim=-1)
#   return uv

# def binarize(x, q):
#   return torch.where(x.gt(x.flatten(start_dim=1).quantile(q=q, dim=1)[:, None, None, None]), 1., 0.)

# def get_edge_probability(SI, quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
#   p_edge = sum(binarize((SI[..., 0, 0] + SI[..., 1, 1]).unsqueeze(-1), q=q) for q in quantiles) / len(quantiles)
#   return p_edge[:, None, :, :, 0]

# def xQxt(Q, x):
#   Qx = torch.matmul(Q, x.unsqueeze(-1))
#   xQx = torch.matmul(x[..., None].transpose(-1, -2), Qx)
#   return xQx

# def tr(Q):
#   return Q[..., 0, 0] + Q[..., 1, 1]

# def structure_tensor(I, inner_gaussian, left_diff, right_diff, outer_gaussian):
#   # Compute spatial derivatives
#   I = filter_separable_2d_with_reflect_pad(I, inner_gaussian)
#   Ixl = filter_2d_with_reflect_pad(I, left_diff)
#   Iyl = filter_2d_with_reflect_pad(I, left_diff.T)
#   Ixr = filter_2d_with_reflect_pad(I, right_diff)
#   Iyr = filter_2d_with_reflect_pad(I, right_diff.T)

#   # Structure tensor components
#   IxIx = Ixl*Ixl + Ixr*Ixr
#   IxIy = Ixl*Iyl + Ixr*Iyr
#   IyIy = Iyl*Iyl + Iyr*Iyr

#   # Outer blur
#   IxIx = filter_separable_2d_with_reflect_pad(IxIx, outer_gaussian)
#   IxIy = filter_separable_2d_with_reflect_pad(IxIy, outer_gaussian)
#   IyIy = filter_separable_2d_with_reflect_pad(IyIy, outer_gaussian)

#   # Keep only one channel
#   IxIx, IxIy, IyIy = IxIx.mean(1), IxIy.mean(1), IyIy.mean(1)

#   # Structure tensor matrix
#   SI = torch.stack([IxIx, IxIy, IxIy, IyIy], dim=-1).view(*IxIx.shape, 2, 2)

#   return SI

def filter_separable_2d_with_reflect_pad(x, h):
  x = filter_2d_with_reflect_pad(x, h[None, :])
  x = filter_2d_with_reflect_pad(x, h[:, None])
  return x

def filter_2d_with_reflect_pad(x, h):
  _, C, _, _ = x.shape
  p2d = tuple([(h.size(1)-1)//2] * 2 + [(h.size(0)-1)//2] * 2)
  x = F.pad(x, p2d, "reflect")
  return F.conv2d(x, h[None, None, :, :].expand((C, -1, -1, -1)), padding='valid', groups=C)

def binarize(x, q):
  return torch.where(x.gt(x.flatten(start_dim=1).quantile(q=q, dim=1)[:, None, None, None]), 1., 0.)

def make_gaussian_filter(stddev):
  order = math.ceil(3*stddev)
  n = np.arange(-order, order+1)
  h = math.exp(-stddev) * scipy.special.iv(n, stddev)
  h = np.array(h, dtype=np.float32)
  h = h / np.sum(h)
  return h