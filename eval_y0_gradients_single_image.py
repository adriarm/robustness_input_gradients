import argparse
import logging
import math
import os
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

import gdown
import numpy as np
import scipy
# timm func
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models import create_model
from timm.utils import AverageMeter, accuracy, reduce_tensor
from torchvision import transforms

from data.dataset import ImageNet
# from model.resnet_denoise import get_FD
from model import vit_mae
from model.model_zoo import model_zoo
from model.resnet import resnet50, wide_resnet50_2
from utils import (NormalizeByChannelMeanStd, create_logger, distributed_init,
                   random_seed)

def get_model(model_name, ckpt_path=''):
    backbone=model_zoo[model_name]['model']
    url = model_zoo[model_name]['url']

    src_path='./src_ckpt'
    ckpt_name=f'{model_name}_checkpoint.pth'
    ckpt_dir=os.path.join(src_path, ckpt_name)
    ckpt_list=os.listdir(src_path)
    if (ckpt_path == '') and (ckpt_name not in ckpt_list):
        gdown.download(url, ckpt_dir, quiet=False)
    
    mean=model_zoo[model_name]['mean']
    std=model_zoo[model_name]['std']
    pretrained=model_zoo[model_name]['pretrained']
    act_gelu=model_zoo[model_name]['act_gelu']
    
    if backbone=='resnet50_rl':
        model=resnet50()
    elif backbone=='wide_resnet50_2_rl':
        model=wide_resnet50_2()
    # elif backbone=='resnet152_fd':
    #     model = get_FD()
    elif backbone=='vit_base_patch16' or backbone=='vit_large_patch16':
        model=vit_mae.__dict__[backbone](num_classes=1000, global_pool='')
    else:
        model_kwargs=dict({'num_classes': 1000})
        if act_gelu:
            model_kwargs['act_layer']=nn.GELU
        model = create_model(backbone, pretrained=pretrained, **model_kwargs)
    
    if not pretrained:
        ckpt=torch.load(ckpt_dir, map_location='cpu')
        model.load_state_dict(ckpt)
    if ckpt_path != '':
        ckpt=torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict_ema'])

    normalize = NormalizeByChannelMeanStd(mean=mean, std=std)
    model = torch.nn.Sequential(normalize, model)

    return model

def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)

    # local test parameters
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend used to set up distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Model parameters
    parser.add_argument('--model_name', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--ckpt-path', default='', type=str, metavar='MODEL', help='Path of model')
    parser.add_argument('--num-classes', default=1000, type=int, help='number of classes')

    # data parameters
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--batch_size', default=25, type=int)
    parser.add_argument('--crop-pct', default=0.875, type=float, metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', default=3, type=int, help='1: lanczos 2: bilinear 3: bicubic')
    parser.add_argument('--imagenet_val_path', default='/imagenet/val/', type=str, help='path to imagenet validation dataset')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin_mem', default=True)

    # feature gradients
    parser.add_argument('--mode', type=str, default='eval')
    parser.add_argument('--idx', type=int, default=1, help='image idx')

    # gradient teacher
    parser.add_argument('--sigma', default=1., type=float)
    parser.add_argument('--k_sobel', default=3, type=int)
    parser.add_argument('--a', default=.7, type=float)

    # output
    parser.add_argument('--output-dir', default='./test_out', type=str)
    
    return parser

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

def main(args):
    #distributed settings
    if "WORLD_SIZE" in os.environ:
        args.world_size=int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ:
        args.local_rank=int(os.environ["LOCAL_RANK"])
    args.distributed=True
    assert args.world_size == 1, "Do with only GPU, don't want to deal with this"
    distributed_init(args)

    # get model
    model = get_model(args.model_name, args.ckpt_path).cuda()
    # if args.distributed:
    #     model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.device_id], find_unused_parameters=True)
    print(model)    
    
    # Output dir and logger
    if hasattr(model[1], 'teacher_a'):
        args.output_dir = f'{args.output_dir}/a_{model[1].teacher_a}'
    args.output_dir = f'{args.output_dir}/{args.mode}'
    _logger = create_logger(args.output_dir, dist_rank=args.rank, name='main_benchmark', default_level=logging.INFO)

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark = True

    # test transform without norm
    meta_file='val_grads' if 'val' in args.imagenet_val_path else 'train_grads'
    dataloader_eval, dataset_eval = get_dataloader(args, root=args.imagenet_val_path, meta_file=meta_file)
    
    # Clean acc for debugging
    # clean_acc=validate(model, dataloader_eval, args, _logger)
    # _logger.info('Top1 acc of clean images is: {0:>7.4f}'.format(clean_acc['top1']))

    # # Measure gradients
    # _logger.info('Module name list')
    # _logger.info(list(dict(model.named_modules()).keys()))

    for i in range(len(dataset_eval)):
        args.idx = i
        images = visualize_y0_gradient(model, dataloader_eval, args, _logger)
        torchvision.utils.save_image(images, Path(args.output_dir) / f'images_{meta_file}_{args.idx}.png', nrow=9, normalize=True, scale_each=True)    

def visualize_y0_gradient(model, loader, args, _logger, log_suffix='layer gradient'):
    batch_time_m = AverageMeter()

    if args.mode == 'eval':
        model.eval()
    elif args.mode == 'train':
        model.train()
    else:
        assert False, f'args.mode should be train or eval; is {args.mode}'

    # Teacher
    teacher_args = {
      'sigma':args.sigma,
      'k_sobel':args.k_sobel,
      'a':args.a,
    }
    contour_energy = ContourEnergy(**teacher_args).cuda()
    
    # Select image
    input, target = next(iter(loader))
    input, target = input[args.idx:args.idx+1], target[args.idx:args.idx+1]
    
    # Send to gpu
    input = input.cuda()
    target = target.cuda()

    # Forward
    input.requires_grad_(True)
    output = model(input)
    loss = torch.nn.functional.cross_entropy(output, target)
    
    # Backward
    loss_gradient = torch.autograd.grad(loss, input, create_graph=False, retain_graph=True)[0]
    logit_gradient = torch.autograd.grad(output[torch.arange(output.size(0)), target], input, create_graph=False, retain_graph=True)[0]

    # Print reg losses
    meta='val' if 'val' in args.imagenet_val_path else 'train'

    # Prepare images
    _, N, H, W = input.shape
    loss_gradient_rgb = loss_gradient.view(N, 1, H, W).expand(-1, N, -1, -1)
    logit_gradient_rgb = logit_gradient.view(N, 1, H, W).expand(-1, N, -1, -1)

    images = torch.concat([input, loss_gradient, loss_gradient_rgb, logit_gradient, logit_gradient_rgb])
    images = torch.concat([images, contour_energy(images).expand_as(images)])

    images = abs_normalize(images, q=0.01)        

    return images

def abs_normalize(x, q=None, start_dim=-3):
  s = torch.quantile(x.abs().flatten(start_dim=start_dim), q=max(q, 1-q), dim=-1, keepdim=False)
  x = 0.5 + 0.5 * x/s[(..., ) + (None,)*(-start_dim)]

  x = torch.clamp(x, 0., 1.)
  return x

def validate(model, loader, args, _logger, log_suffix='clean acc'):
    batch_time_m = AverageMeter()
    top1_m = AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():
            output = model(input)

        acc1, _ = accuracy(output, target, topk=(1, 5))

        if args.distributed:
            acc1 = reduce_tensor(acc1, args.world_size)

        torch.cuda.synchronize()

        top1_m.update(acc1.item(), input.size(0))

        batch_time_m.update(time.time() - end)
        end = time.time()

        log_name = 'Test ' + log_suffix
        _logger.info(
            '{0}: [{1:>4d}/{2}]  '
            'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
            'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '.format(
                log_name, batch_idx, last_idx, batch_time=batch_time_m, top1=top1_m))

    metrics = OrderedDict([('top1', top1_m.avg)])

    return metrics


class ContourEnergy(nn.Module):
  def __init__(self, sigma, k_sobel, a) -> None:
    super().__init__()
    self.sigma = sigma
    self.k_sobel = k_sobel
    self.a = a

    # Gaussian kernel
    gaussian_kernel = make_gaussian_filter(sigma)
    self.register_buffer('gaussian_kernel', torch.from_numpy(gaussian_kernel), persistent=False)

    # Sobel kernel
    sobel_kernel = make_sobel_kernel(k_sobel)
    self.register_buffer('sobel_kernel', torch.from_numpy(sobel_kernel), persistent=False)

  def forward(self, img):
    N, C, H, W = img.shape
    
    # Blur
    p2d = tuple([(self.gaussian_kernel.size(0)-1)//2] * 4)
    img = F.pad(img, p2d, "reflect")
    img = F.conv2d(img, self.gaussian_kernel[None, None, :, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    img = F.conv2d(img, self.gaussian_kernel[None, None, None, :].expand((C, -1, -1, -1)), padding='valid', groups=C)
    
    # Sobel
    p2d = tuple([(self.sobel_kernel.size(0)-1)//2] * 4)
    img = F.pad(img, p2d, "reflect")
    dx = F.conv2d(img, self.sobel_kernel[None, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    dy = F.conv2d(img, self.sobel_kernel.T[None, None].expand((C, -1, -1, -1)), padding='valid', groups=C)
    
    # Square and sum
    energy = (dx**2 + dy**2).sum(-3, keepdim=True)**0.5

    # Blur (for centering)
    p2d = tuple([(self.gaussian_kernel.size(0)-1)//2] * 4)
    energy = F.pad(energy, p2d, "reflect")
    energy = F.conv2d(energy, self.gaussian_kernel[None, None, :, None], padding='valid')
    energy = F.conv2d(energy, self.gaussian_kernel[None, None, None, :], padding='valid')

    # Weight
    t_img = energy #1 - torch.exp(-self.a * energy**2)
    
    return t_img

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

def greyscale_functional(x):
  return x.mean(-3, keepdim=True).expand_as(x)

class ToGreyscale(nn.Module):
  def __init__(self) -> None:
    super().__init__()

  def forward(self, x):
    return greyscale_functional(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust test script', parents=[get_args_parser()])
    args = parser.parse_args()

    main(args)