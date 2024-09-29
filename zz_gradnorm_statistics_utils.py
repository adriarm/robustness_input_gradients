import torch
import torch.nn as nn
import timm
import numpy as np
import torchvision
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# import robustbench

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import os
import plotly.express as px
# import skimage

from tqdm.auto import tqdm

from torchvision import transforms
import argparse
from data.dataset import ImageNet

#import torchattacks

# Plotting
def abs_normalize(x, q=None, start_dim=-3):
  s = torch.quantile(x.abs().flatten(start_dim=start_dim), q=max(q, 1-q), dim=-1, keepdim=False)
  x = 0.5 + 0.5 * x/s[(..., ) + (None,)*(-start_dim)]

  x = torch.clamp(x, 0., 1.)
  return x
  
def plot_side_by_side_normalize(*images, normalize, title=None, cmap='gray', plot_numbers=False):
  fig = plt.figure(figsize=(20,20))
  if title is not None:
    fig.suptitle(title, fontsize=16)
  assert len(images) == len(normalize)
  columns = len(images)
  for i, image in enumerate(images):
      ax = plt.subplot(len(images) // columns + 1, columns, i + 1)
      image = image.detach()
      with torch.no_grad():
        if image.shape[-1] > 3:
          image = image.permute(1, 2, 0)
        image_n = image
        if normalize[i]:
          image_n = abs_normalize(image_n, q=0.01)
        plt.imshow(image_n, cmap=cmap)
        if plot_numbers:
          assert image.shape[-1] == 1
          for (j,i),label in np.ndenumerate(image[..., 0]):
            ax.text(i,j,round(label, 2),ha='center',va='center')
  fig.tight_layout()
  fig.subplots_adjust(top=1.20)
  plt.show()

def select_nonzero(x):
  assert len(x.shape) == 3
  nz_idx = x.nonzero()
  i0 = nz_idx[:, 1].min()
  i1 = nz_idx[:, 1].max()
  j0 = nz_idx[:, 2].min()
  j1 = nz_idx[:, 2].max()
  return x[:, i0:i1+1, j0:j1+1]

# Hooks
from typing import Callable, Any

def get_output(module, input, output):
    return output

def get_input(module, input, output):
    return input

def get_input_output(module, input, output):
    return input, output

def get_module_input_output(module, input, output):
    return module, input, output

def register_hook_fn_to_module(model: nn.Module, module_name: str, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        if module_name == name:
            handle = m.register_forward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
            return handle, results_dict

def register_bkw_hook_fn_to_module(model: nn.Module, module_name: str, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        if module_name == name:
            handle = m.register_full_backward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
            return handle, results_dict

def register_hook_fn_to_all_modules(model: nn.Module, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        _ = m.register_forward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
    return results_dict

def register_hook_fn_to_all_childless_modules(model: nn.Module, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    handles = {}
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            handles[name] = m.register_forward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
    return handles, results_dict

def register_bkw_hook_fn_to_all_modules(model: nn.Module, hook_fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], Any]):
    results_dict = {}
    for name, m in model.named_modules():
        _ = m.register_full_backward_hook(_hook_fn_cntr(name, results_dict, hook_fn))
    return results_dict

def _hook_fn_cntr(name, activation_dict, hook_fn):
    def hook(model, input, output):
        activation_dict[name] = hook_fn(model, input, output)
    return hook

# Model

def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)
            
        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)

def add_imagenet_normalization(model):
  val_transform = create_transform(
    **resolve_data_config(model.pretrained_cfg, model=model),
    is_training=False,
  )

  normalize_transform = val_transform.transforms[-1]
  model = nn.Sequential(normalize_transform, model)
  return model

def load_model(path, ema=False):
    model_kwargs=dict({
            'num_classes': 1000,
            'drop_rate': 0.0,
            'drop_path_rate': 0.0,
            'drop_block_rate': None,
            'global_pool': None,
            'bn_momentum': None,
            'bn_eps': None,
    })
    print(path, f'ema={ema}')
    if '_resnet_' in path:
      model = timm.models.create_model('resnet50', pretrained=False, **model_kwargs)
      if '_gelu' in path:
        replace_layers(model, nn.ReLU, nn.GELU())
    elif '_swinb' in path:
      model = timm.models.create_model('swin_base_patch4_window7_224', pretrained=False, **model_kwargs)
    elif '_swins' in path:
      model = timm.models.create_model('swin_small_patch4_window7_224', pretrained=False, **model_kwargs)
    ckpt = torch.load(path, map_location='cpu')
    if ema:
      if 'state_dict_ema' in ckpt:
        model.load_state_dict(ckpt['state_dict_ema'])
      else:
        return None
    else:
      model.load_state_dict(ckpt['state_dict'])
    model.load_state_dict(ckpt['state_dict'])

    model = add_imagenet_normalization(model)
    return model.eval().cpu()

# def load_public_model(model_name):
#   if model_name[0].isupper():
#     return robustbench.utils.load_model(model_name, dataset='imagenet', threat_model='Linf').eval().to(device)
#   else:
#     if 'random' in model_name:
#       return add_imagenet_normalization(timm.create_model(model_name[:-len('_random')], pretrained=False)).eval().cpu()
#     else:
#       return add_imagenet_normalization(timm.create_model(model_name, pretrained=True)).eval().cpu()

# Data
def get_data(N=1000, batch_size=16):
  data = torch.load('/var/datasets/adrianr/input_norm/analysis_data/240206_gen_imagenet_data_10k_noattack.pth', map_location='cpu')

  xs = data['xs']
  ys = data['ys']

  xs = xs[:N]
  ys = ys[:N]

  model = timm.create_model('resnet50', pretrained=True)

  val_transform = create_transform(
      **resolve_data_config(model.pretrained_cfg, model=model),
      is_training=False,
    )

  normalize_transform = val_transform.transforms[-1]

  if xs.min() < 0:
    xs = xs*normalize_transform.std[None, :, None, None] + normalize_transform.mean[None, :, None, None]

  sampler_indices = range(N)
  ds = torch.utils.data.TensorDataset(xs, ys)

  dataloader = torch.utils.data.DataLoader(
      ds,
      batch_size=batch_size,
      sampler=sampler_indices,
  )

  return xs, ys, ds, dataloader

## Viz data
def get_dataloader(args, root, meta_file, batch_size=None):
    t = []
    interpolation = args.interpolation
    if isinstance(interpolation, str):
        if interpolation == 'lanczos': interpolation = 1
        if interpolation == 'bilinear': interpolation = 2
        if interpolation == 'bicubic': interpolation = 3
    if args.input_size > 32:
        size = int(args.input_size/args.crop_pct)
        t.append(
            transforms.Resize(size, interpolation=interpolation),
        )
        t.append(transforms.CenterCrop(args.input_size))
    else:
        t.append(
            transforms.Resize(args.input_size, interpolation=interpolation),
        )
    t.append(transforms.ToTensor())
    # t.append(ToGreyscale())
    test_transform = transforms.Compose(t)

    # set dataloader
    # dataset_eval=ImageNet(root=args.imagenet_val_path, meta_file='./data/imagenet_val_1k.txt', transform=test_transform)
    # args.imagenet_val_path='./src_data/ILSVRC2012_img_val'
    if 'SLURM_PROCID' in os.environ:
        cmd = os.popen('modulecmd python load "/home/gridsan/groups/datasets/ImageNet/modulefile"')
        cmd.read()
        cmd.close()
        #_logger.info(f'Imagenet path {os.environ["IMAGENET_PATH"]}')
        args.imagenet_val_path = '/run/user/61863/imagenet' + '/normal/val'
    dataset_eval=ImageNet(root=root, meta_file=f'./src_data/{meta_file}.txt', transform=test_transform)
    sampler_eval=None
    # if args.distributed:
    #     sampler_eval = torch.utils.data.distributed.DistributedSampler(dataset_eval)
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size if batch_size is None else batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    return dataloader_eval, dataset_eval

def get_viz_fig_data(batch_size):
    args = {
      'input_size':224,
      'crop_pct':0.875,
      'interpolation':'bicubic',
      'num_workers':4,
      'pin_mem':True
    }
    args = argparse.Namespace(**args)

    data_path = '/data/vision/torralba/datasets/imagenet_pytorch/ImageNet/train'
    meta_file = 'train_grads'
    dataloader, dataset = get_dataloader(args, root=data_path, meta_file=meta_file, batch_size=batch_size)
    xs = torch.cat([xy[0] for xy in dataloader], 0)
    ys = torch.cat([xy[1] for xy in dataloader], 0)
    return xs, ys, dataset, dataloader

# Statistics
def get_statistics(model, dataloader, device='cpu'):
  model = model.to(device)

  # handle, act_dict = register_hook_fn_to_module(model, module_name, get_input_output)
  
  out_list = []
  loss_list = []

  grad_loss_x_list = []
  grad_class_x_list = []

  for ii, (x, y) in enumerate(tqdm(dataloader)):
      x, y = x.to(device), y.to(device)
      x.requires_grad_(True)

      # Output and loss
      out = model(x)#.cpu()
      out_list.append(out.detach().cpu())

      loss_v = torch.nn.functional.cross_entropy(out, y, reduction='none')      
      loss_list.append(loss_v.detach().cpu())


      # # Activation
      # act_input = act_dict[module_name][0][0]
      # act_input_list.append(act_input.detach().cpu())

      # act_output = act_dict[module_name][1][0] if isinstance(act_dict[module_name][1], tuple) else act_dict[module_name][1]
      # act_output_list.append(act_output.detach().cpu())


      # Loss gradients
      loss = loss_v.sum()

      grad_loss_x = torch.autograd.grad(loss, x, create_graph=False, retain_graph=True)[0].detach().cpu() #.abs().sum(1, keepdim=True)
      grad_loss_x_list.append(grad_loss_x)

      grad_class_x = torch.autograd.grad(out[torch.arange(out.size(0)), y].sum(), x, create_graph=False, retain_graph=True)[0].detach().cpu() #.abs().sum(1, keepdim=True)
      grad_class_x_list.append(grad_class_x)
  
  out = torch.cat(out_list, 0)
  loss = torch.cat(loss_list, 0)

  grad_loss_x = torch.cat(grad_loss_x_list, 0)
  grad_class_x = torch.cat(grad_class_x_list, 0)
  
  # handle.remove()
  model = model.to('cpu')
  model = None

  stats = {
    'out':out,
    'loss':loss,
    'grad_loss_x':grad_loss_x,
    'grad_class_x':grad_class_x,
  }

  return stats

import torchattacks
def get_attack_statistics(model, dataloader, device='cpu'):
  model = model.to(device)

  # handle, act_dict = register_hook_fn_to_module(model, module_name, get_input_output)

  atk_list = []  
  out_list = []
  loss_list = []

  grad_loss_x_list = []
  grad_class_x_list = []

  atk = torchattacks.PGD(model, eps=4./255., steps=10, random_start=True)
  atk_name = 'pgd10-4./255.-rs'

  for ii, (x, y) in enumerate(tqdm(dataloader)):
      x, y = x.to(device), y.to(device)

      x = atk(x, y)
      x.requires_grad_(True)
      atk_list.append(x.detach().cpu())

      # Output and loss
      out = model(x)#.cpu()
      out_list.append(out.detach().cpu())

      loss_v = torch.nn.functional.cross_entropy(out, y, reduction='none')      
      loss_list.append(loss_v.detach().cpu())


      # # Activation
      # act_input = act_dict[module_name][0][0]
      # act_input_list.append(act_input.detach().cpu())

      # act_output = act_dict[module_name][1][0] if isinstance(act_dict[module_name][1], tuple) else act_dict[module_name][1]
      # act_output_list.append(act_output.detach().cpu())


      # Loss gradients
      loss = loss_v.sum()

      grad_loss_x = torch.autograd.grad(loss, x, create_graph=False, retain_graph=True)[0].detach().cpu() #.abs().sum(1, keepdim=True)
      grad_loss_x_list.append(grad_loss_x)

      grad_class_x = torch.autograd.grad(out[torch.arange(out.size(0)), y].sum(), x, create_graph=False, retain_graph=True)[0].detach().cpu() #.abs().sum(1, keepdim=True)
      grad_class_x_list.append(grad_class_x)
  
  atk = torch.cat(atk_list, 0)
  out = torch.cat(out_list, 0)
  loss = torch.cat(loss_list, 0)

  grad_loss_x = torch.cat(grad_loss_x_list, 0)
  grad_class_x = torch.cat(grad_class_x_list, 0)
  
  # handle.remove()
  model = model.to('cpu')
  model = None

  stats = {
    'atk_name':atk_name,
    'atk':atk,
    'out':out,
    'loss':loss,
    'grad_loss_x':grad_loss_x,
    'grad_class_x':grad_class_x,
  }

  return stats

def get_model_statistics(model_path, ema, dataloader, device='cpu'):
  stats_dir = f'{model_path.split(".")[0]}_arxiv_stats'
  os.makedirs(stats_dir, exist_ok=True)
  stats_path = f'{stats_dir}/stats.pth.tar'
  atk_stats_path = f'{stats_dir}/atk_stats.pth.tar'

  if not os.path.isfile(stats_path):
    model = load_model(model_path, ema=ema)
    stats = get_statistics(model, dataloader, device=device)
    torch.save(stats, f=stats_path)
  else:
    stats = torch.load(stats_path)
  
  # if not os.path.isfile(atk_stats_path):
  #   model = load_model(model_path, ema=ema)
  #   atk_stats = get_attack_statistics(model, dataloader, device=device)
  #   torch.save(atk_stats, f=atk_stats_path)
  # else:
  #   atk_stats = torch.load(atk_stats_path)
  
  return stats #, atk_stats

def get_model_atk_statistics(model_path, ema, dataloader, device='cpu'):
  stats_dir = f'{model_path.split(".")[0]}_arxiv_stats'
  os.makedirs(stats_dir, exist_ok=True)
  stats_path = f'{stats_dir}/stats.pth.tar'
  atk_stats_path = f'{stats_dir}/atk_stats.pth.tar'

  if not os.path.isfile(atk_stats_path):
    model = load_model(model_path, ema=ema)
    atk_stats = get_attack_statistics(model, dataloader, device=device)
    torch.save(atk_stats, f=atk_stats_path)
  else:
    atk_stats = torch.load(atk_stats_path)
  
  return atk_stats

# ## Spectral energy
# def _shape_check(x, shape):
#   ### Assert ndim
#   assert len(x.shape) == len(shape), f'Array must be {len(shape)}-dimensional'

#   ### Assert shape
#   for d, (a, b) in enumerate(zip(x.shape, shape)):
#     if b is not None:
#       assert a == b, f'Shape does not match at dimension {d}; array shape: {x.shape}; target shape: {shape}'

# def get_spectral_energy(weight, N=512):
#   fft = torch.fft.fft2(weight, s=[N, N], dim=(-2, -1)).abs()
#   fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
#   fft_shift = fft_shift.cpu().numpy()**2

#   return fft_shift

# def get_spectral_phase(weight, N=512):
#   fft = torch.fft.fft2(weight, s=[N, N], dim=(-2, -1)).angle()
#   fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
#   fft_shift = fft_shift.cpu().numpy() #**2

#   return fft_shift

# def get_polar_marginal_spectral_energy(weight, N=512):
#   # Get dimensions
#   _shape_check(weight, (None, None, None, None))
#   out_channels, in_channels, _, _ = weight.shape

#   # Calculate FFT
#   fft = torch.fft.fft2(weight, s=[N, N], dim=(-2, -1)).abs()
#   fft_shift = torch.fft.fftshift(fft, dim=(-2, -1))
#   fft_shift = fft_shift.cpu().numpy()
  
#   # Radial transform
#   fft_shift = fft_shift.reshape(out_channels * in_channels, N, N)
#   fft_polar = skimage.transform.warp_polar(fft_shift, radius=N//2, channel_axis=0)
  
#   # Integrate
#   fft_radial = ((fft_polar**2).sum(1)) #/ (np.pi*np.arange(1, N//2+1)**2)
#   fft_radial= fft_radial.reshape(out_channels, in_channels, N//2)

#   fft_angular = ((fft_polar**2).sum(2))
#   fft_angular = fft_angular.reshape(out_channels, in_channels, 360)

#   return fft_radial, fft_angular

# def plot_2d_spectrals(*ws, log_scale=False):
#   ffts = np.stack([get_spectral_energy(w) for w in ws], 0)
#   ffts = ffts / ffts.sum((1,2,3), keepdims=True)
#   if log_scale:
#     ffts = np.log10(ffts)
#   ffts = ffts.transpose(0, 2, 3, 1)
#   # print(ffts.shape)
#   # ffts_radial_angular = [get_polar_marginal_spectral_energy(w) for w in ws]

#   fig = px.imshow(img=ffts[..., 0], facet_col=0, color_continuous_scale='jet')
#   for i in range(len(ws)):
#     fig.for_each_annotation(lambda a: a.update(text=a.text.replace("facet_col={i}", "")))
#   plt.tight_layout()
#   fig.show()

# def plot_2d_phase(*ws):
#   ffts_phase = np.stack([get_spectral_phase(w) for w in ws], 0)
#   ffts_spectral = np.stack([get_spectral_energy(w) for w in ws], 0)
#   ffts_spectral = ffts_spectral / ffts_spectral.sum((1,2,3), keepdims=True)
#   ffts = ffts_phase * ffts_spectral
#   ffts = ffts.transpose(0, 2, 3, 1)
#   # print(ffts.shape)
#   # ffts_radial_angular = [get_polar_marginal_spectral_energy(w) for w in ws]

#   fig = px.imshow(img=ffts[..., 0], facet_col=0, color_continuous_scale='jet')
#   for i in range(len(ws)):
#     fig.for_each_annotation(lambda a: a.update(text=a.text.replace("facet_col={i}", "")))
#   plt.tight_layout()
#   fig.show()

# def plot_radial_spectrals(*ws):
#   #print(ws[0].shape)
#   #print(get_polar_marginal_spectral_energy(ws[0][None, :, :, :])[0].shape)
#   ffts_radial = np.stack([get_polar_marginal_spectral_energy(w[None, :, :, :])[0] for w in ws], axis=0)
#   #print(ffts_radial.shape)
#   ffts_radial = ffts_radial / np.mean(ffts_radial, axis=(2, 3), keepdims=True)
#   ffts_radial = ffts_radial.sum((1, 2))
#   #print(ffts_radial.shape)

#   # Frequency sweep
#   N = ffts_radial.shape[-1] * 2
#   x = np.arange(N//2) / N

#   data = np.concatenate([x[None], ffts_radial], 0)
#   # print(data.shape)
#   df = pd.DataFrame(data=data.T)

#   # Plot
#   fig = px.line(data_frame=df, x=0, y=df.columns,
#     title='Radial Marginal Spectral Energy',
#     labels={
#       "x": "|F|",
#       "value": "Energy",
#     },
#   )
#   fig.show()

# def plot_angular_spectrals(*ws):
#   #print(ws[0].shape)
#   #print(get_polar_marginal_spectral_energy(ws[0][None, :, :, :])[0].shape)
#   ffts_angular = np.stack([get_polar_marginal_spectral_energy(w[None, :, :, :])[1] for w in ws], axis=0)
#   #print(ffts_radial.shape)
#   ffts_angular = ffts_angular / np.mean(ffts_angular, axis=(2, 3), keepdims=True)
#   ffts_angular = ffts_angular.sum((1, 2))
#   # print(ffts_angular.shape)

#   # Frequency sweep
#   theta = np.arange(360)

#   data = np.concatenate([theta[None], ffts_angular], 0)
#   # print(data.shape)
#   df = pd.DataFrame(data=data.T)

#   # Plot
#   # fig = px.line_polar(data_frame=df, theta=0, r=[1, 2],
#   #   line_close=True,
#   #   title='Angular Marginal Spectral Energy',
#   #   direction='counterclockwise',
#   #   start_angle=0
#   # )

#   fig = px.line(data_frame=df, x=0, y=df.columns,
#     title='Angular Marginal Spectral Energy',
#     labels={
#       "x": "|F|",
#       "value": "Energy",
#     },
#   )
#   fig.show()

if __name__ == '__main__':
  device = torch.device('cuda', 6)
  xs, ys, ds, dataloader = get_data(N=10000, batch_size=32)

  # stats_at, atk_stats_at = get_model_statistics(f'outputs/advtrain_swinb_orig/last.pth.tar', ema=True, dataloader=dataloader, device=device)
  stats_gn14, atk_stats_gn14 = get_model_statistics(f'arxiv_outputs/gradnorm_swinb_finetuning_pareto_14/2024-04-03_18-58-08/last.pth.tar', ema=True, dataloader=dataloader, device=device)



