# Code adapted from: https://github.com/thu-ml/adversarial_training_imagenet
# @article{liu2023comprehensive,
#   title={A Comprehensive Study on Robustness of Image Classification Models: Benchmarking and Rethinking},
#   author={Liu, Chang and Dong, Yinpeng and Xiang, Wenzhao and Yang, Xiao and Su, Hang and Zhu, Jun and Chen, Yuefeng and He, Yuan and Xue, Hui and Zheng, Shibao},
#   journal={arXiv preprint arXiv:2302.14301},
#   year={2023}
# }

import warnings

warnings.filterwarnings("ignore")
import argparse
import copy
import logging
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.io import savemat
# timm functions
from timm.models import load_checkpoint, model_parameters, resume_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import (AverageMeter, CheckpointSaver, ModelEmaV2, accuracy,
                        dispatch_clip_grad, distribute_bn, get_outdir,
                        reduce_tensor, update_summary)
from torch.nn.parallel import DistributedDataParallel as NativeDDP

import input_norm_losses
from adv.adv_utils import adv_generator
from data.dataset import build_dataset
from eval_y0_gradients_single_image import abs_normalize
from eval_y0_gradients_single_image import \
    get_dataloader as get_dataloader_for_visualization
# gradient teachers
from gradient_teachers import ContourEnergy
from model.loss import build_loss, build_loss_scaler, resolve_amp
from model.model import ToGreyscale, build_model
# in functions
from utils import (create_logger, distributed_init, formatted_array_str,
                   get_flattened_gradients, random_seed)

from timm.utils.model import get_state_dict, unwrap_model

import numpy as np
np.set_printoptions(threshold=np.inf)

import swin_transformer_timm_version

# torch.autograd.set_detect_anomaly(True)


def get_args_parser():
    parser = argparse.ArgumentParser('Robust training script', add_help=False)
    parser.add_argument('--configs', default='', type=str)

    #* distributed setting
    parser.add_argument('--distributed', default=True)
    parser.add_argument('--local-rank', default=-1, type=int)
    parser.add_argument('--device-id', type=int, default=0)
    parser.add_argument('--rank', default=-1, type=int, help='rank')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-backend', default='nccl', help='backend used to set up distributed training')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    
    #* amp parameters
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--amp_version', default='', help='amp version')

    #* model parameters
    parser.add_argument('--model', default='resnet50', type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--replace_relu_with_gelu', default=False, help='if use GELU')
    parser.add_argument('--replace_relu_with_silu', default=False, help='if use SiLU')
    parser.add_argument('--relu_not_inplace', default=False, help='if switch to not inplace ReLU')
    parser.add_argument('--num-classes', default=1000, type=int, help='number of classes')
    parser.add_argument('--create_model_pretrained', default=True, help='Create model pretrained')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain', default='', help='pretrain from checkpoint')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None. (opt)')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout (opt)')

    #* Batch norm parameters
    parser.add_argument('--bn-momentum', type=float, default=None, help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None, help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true', default=False, help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce', help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true', default=False,
                        help='Enable separate BN layers per augmentation split.')

    #* Optimizer parameters
    parser.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 0.0001)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                    help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    #* Learning rate schedule parameters
    parser.add_argument('--epochs', default=150, type=int)
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lrb', type=float, default=0.1, metavar='LR',
                        help='base learning rate (default: 5e-4)')
    parser.add_argument('--lr', type=float, default=None, help='actual learning rate')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    #* dataset parameters
    parser.add_argument('--batch-size', default=64, type=int)    # batch size per gpu
    parser.add_argument('--grad-accum', default=1, type=int)    # gradient acumulation
    parser.add_argument('--train-dir', default='', type=str, help='train dataset path')
    parser.add_argument('--eval-dir', default='', type=str, help='validation dataset path')
    parser.add_argument('--input-size', default=224, type=int, help='images input size')
    parser.add_argument('--crop-pct', default=0.875, type=float,
                        metavar='N', help='Input image center crop percent (for validation only)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--mean', type=float, nargs='+', default=(0.485, 0.456, 0.406), metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=(0.229, 0.224, 0.225), metavar='STD',
                        help='Override std deviation of of dataset')
    
    #* Augmentation & regularization parameters
    parser.add_argument('--no-aug', action='store_true', default=False,
                        help='Disable all training augmentation, override other train aug args')
    parser.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                        help='Random resize scale (default: 0.08 1.0)')
    parser.add_argument('--ratio', type=float, nargs='+', default=[3./4., 4./3.], metavar='RATIO',
                        help='Random resize aspect ratio (default: 0.75 1.33)')
    parser.add_argument('--hflip', type=float, default=0.5,
                        help='Horizontal flip training aug probability')
    parser.add_argument('--vflip', type=float, default=0.,
                        help='Vertical flip training aug probability')
    parser.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--aa', type=str, default=None, metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". (default: None)'),
    parser.add_argument('--aug-repeats', type=float, default=0,
                        help='Number of augmentation repetitions (distributed training only) (default: 0)')
    parser.add_argument('--aug-splits', type=int, default=0,
                        help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
    parser.add_argument('--jsd-loss', action='store_true', default=False,
                        help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
    parser.add_argument('--bce-loss', action='store_true', default=False,
                        help='Enable BCE loss w/ Mixup/CutMix use.')
    parser.add_argument('--bce-target-thresh', type=float, default=None,
                        help='Threshold for binarizing softened BCE targets (default: None, disabled)')
    # random erase
    parser.add_argument('--reprob', type=float, default=0., metavar='PCT',
                        help='Random erase prob (default: 0.)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')
    parser.add_argument('--mixup', type=float, default=0.0,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix', type=float, default=0.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
    parser.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                        help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')
    parser.add_argument('--train-interpolation', type=str, default='random',
                        help='Training interpolation (random, bilinear, bicubic default: "random")')
    # drop connection
    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    #* ema
    parser.add_argument('--model-ema', action='store_true', default=False,
                        help='Enable tracking moving average of model weights')
    parser.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                        help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
    parser.add_argument('--model-ema-decay', type=float, default=0.9998,
                        help='decay factor for model weights moving average (default: 0.9998)')

    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                    help='how many batches to wait before logging training status')
    parser.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                    help='how many batches to wait before writing recovery checkpoint')
    parser.add_argument('--max-history', type=int, default=5, help='how many recovery checkpoints')
    parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 4)')
    parser.add_argument('--output-dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                        help='Best metric (default: "top1")')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')

    # advtrain
    parser.add_argument('--advtrain', default=False, help='if use advtrain')
    parser.add_argument('--attack-criterion', type=str, default='regular', choices=['regular', 'smooth', 'mixup'], help='default args for: adversarial training')
    parser.add_argument('--attack-eps', type=float, default=4.0/255, help='attack epsilon.')
    parser.add_argument('--attack-step', type=float, default=8.0/255/3, help='attack epsilon.')
    parser.add_argument('--attack-it', type=int, default=3, help='attack iteration')
    # advprop
    parser.add_argument('--advprop', default=False, help='if use advprop')
    # gradnorm
    parser.add_argument('--gradnorm', default=False, help='if use contourtrain')
    parser.add_argument('--alpha', type=float, nargs='+', default=(0.0, 0.1, 1.00), help='Loss weight ramp')
    parser.add_argument('--alpha-start-epoch', type=float, default=0.0, help='Start epoch for loss weight ramp')
    parser.add_argument('--random_start', default=False, help='If use randomstart')
    parser.add_argument('--random_start_eps_factor', type=float, default=0.5, help='Quotient for epsilon')
    # patches
    parser.add_argument('--patch-size', type=int, default=3, help='Patch size for patch similarity')

    # saving
    parser.add_argument('--save_snapshot_for_inference', type=str, nargs='+', default=None, help='When to save snapshots')
    parser.add_argument('--collect_gradient_statistics', type=int, nargs='+', default=None, help='When to collect gradient statistics')

    return parser


def main(args, args_text):
    # distributed settings and logger
    if "WORLD_SIZE" in os.environ:
        args.world_size=int(os.environ["WORLD_SIZE"])
    if "LOCAL_RANK" in os.environ:
        args.local_rank=int(os.environ["LOCAL_RANK"])
    args.distributed=args.world_size>1
    distributed_init(args)

    args.output_dir = f'{args.output_dir}/{os.environ["HYDRA_NOW"]}'
    if args.rank == 0 and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        # os.makedirs(f'{args.output_dir}/gradnorms')
        with open(f'{args.output_dir}/gradnorms.txt', "w") as file:
            print("Gradnorms", file=file)
        os.makedirs(f'{args.output_dir}/snapshots')

    _logger = create_logger(args.output_dir, dist_rank=args.rank, name='main_train', default_level=logging.INFO)

    # fix the seed for reproducibility
    random_seed(args.seed, args.rank)
    torch.backends.cudnn.deterministic=False
    torch.backends.cudnn.benchmark = True
    
    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # resolve amp
    resolve_amp(args, _logger)

    # build model
    model = build_model(args, _logger, num_aug_splits)


    # Replace activations
    def replace_layers(model, old, new):
        for n, module in model.named_children():
            if len(list(module.children())) > 0:
                ## compound module, go inside it
                replace_layers(module, old, new)
                
            if isinstance(module, old):
                ## simple module
                setattr(model, n, new)
    
    if args.replace_relu_with_gelu:
        replace_layers(model, nn.ReLU, nn.GELU())
    if args.replace_relu_with_silu:
        replace_layers(model, nn.ReLU, nn.SiLU())
    if args.relu_not_inplace:
        replace_layers(model, nn.ReLU, nn.ReLU(inplace=False))

    # create optimizer
    optimizer=None
    if args.lr is None:
        args.lr=args.lrb * args.batch_size * args.world_size * args.grad_accum / 512
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

    # build loss scaler
    amp_autocast, loss_scaler = build_loss_scaler(args, _logger)

    # resume from a checkpoint
    resume_epoch = None
    if args.pretrain:
        _ = resume_checkpoint(
            model, args.pretrain,
            optimizer=None,
            loss_scaler=None,
            log_info=args.rank == 0)

    # setup ema
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
        if args.pretrain:
            load_checkpoint(model_ema.module, args.pretrain, use_ema=True)

    # Resume
    if args.resume:
        resume_epoch = resume_checkpoint(
            model, args.resume,
            optimizer=optimizer,
            loss_scaler=loss_scaler,
            log_info=args.rank == 0)
    if args.model_ema:
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)

    # visualize gradients
    # if args.rank == 0:
    #     os.makedirs(Path(args.output_dir) / 'gradient_images')
    # visualize_gradients(args, model, _logger, 'train', epoch='start')
    # visualize_gradients(args, model, _logger, 'val', epoch='start')
    # if model_ema is not None:
    #     visualize_gradients(args, model_ema.module, _logger, 'train', epoch='start', ema='ema')
    #     visualize_gradients(args, model_ema.module, _logger, 'val', epoch='start', ema='ema')
    # visualize_gradients(args, model, _logger, 'train', 0, 0)
    # visualize_gradients(args, model, _logger, 'val', 0, 0)

    # setup distributed training
    if args.distributed:
        if args.amp_version == 'apex':
            # Apex DDP preferred unless native amp is activated
            from apex.parallel import DistributedDataParallel as ApexDDP
            _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[args.device_id])
        # NOTE: EMA model does not need to be wrapped by DDP

    # setup learning rate schedule and starting epoch
    lr_scheduler, num_epochs = create_scheduler(args, optimizer)
    start_epoch = 0
    if resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        lr_scheduler.step(start_epoch)
    _logger.info('Scheduled epochs: {}'.format(num_epochs))

    # create the train and eval dataloaders
    if 'SLURM_PROCID' in os.environ:
        cmd = os.popen('modulecmd python load "/home/gridsan/groups/datasets/ImageNet/modulefile"')
        cmd.read()
        cmd.close()
        #_logger.info(f'Imagenet path {os.environ["IMAGENET_PATH"]}')
        args.train_dir = '/run/user/61863/imagenet' + '/normal/train'
        args.eval_dir = '/run/user/61863/imagenet' + '/normal/val'
    loader_train, loader_eval, mixup_fn = build_dataset(args, num_aug_splits)

    # setup loss function
    train_loss_fn, validate_loss_fn = build_loss(args, mixup_fn, num_aug_splits)

    # setup reg loss fncs
    reg_loss_fn = []
    if args.advtrain:
        reg_loss_fn = None
    elif args.gradnorm and args.loss_fn == 'DBP':
        reg_loss_fn = input_norm_losses.DBP()
    # elif args.gradnorm and args.loss_fn == 'DBPAMHM':
    #     reg_loss_fn = input_norm_losses.DBPAMHM()
    # elif args.gradnorm and args.loss_fn == 'DBPSparsity':
    #     reg_loss_fn = input_norm_losses.DBPSparsity()
    # elif args.gradnorm and args.loss_fn == 'DBPChannel':
    #     reg_loss_fn = input_norm_losses.DBPChannel()
    # elif args.gradnorm and args.loss_fn == 'DBPThresholded':
    #     reg_loss_fn = input_norm_losses.DBPThresholded()
    # elif args.gradnorm and args.loss_fn == 'DBPPow':
    #     reg_loss_fn = input_norm_losses.DBPPow(p=args.p, th=args.th, tol=args.tol)
    # elif args.gradnorm and args.loss_fn == 'DBPTangent':
    #     reg_loss_fn = input_norm_losses.DBPTangent().cuda()
    # elif args.gradnorm and args.loss_fn == 'DBPEdgeWeight':
    #     reg_loss_fn = input_norm_losses.DBPEdgeWeight().cuda()
    # elif args.gradnorm and args.loss_fn == 'DBPEdgeWeightNorm':
    #     reg_loss_fn = input_norm_losses.DBPEdgeWeightNorm().cuda()
    # elif args.gradnorm and args.loss_fn == 'DBPChange':
    #     reg_loss_fn = input_norm_losses.DBPChange().cuda()
    # elif args.gradnorm and args.loss_fn == 'EdgePatchSimilarity':
    #     reg_loss_fn = patch_similarity_losses.EdgePatchSimilarity(patch_size=args.patch_size).cuda()
    _logger.info(f'Reg losses: {str(reg_loss_fn)}')

    # saver
    eval_metric = args.eval_metric
    saver = None
    best_metric = None
    best_epoch = None
    output_dir = None
    if args.rank == 0:
        output_dir = get_outdir(args.output_dir)
        decreasing=True if (eval_metric=='loss' or eval_metric=='advloss') else False
        saver = CheckpointSaver(
            model=model, optimizer=optimizer, args=args, model_ema=model_ema, amp_scaler=loss_scaler,
            checkpoint_dir=output_dir, recovery_dir=output_dir, decreasing=decreasing, max_history=args.max_history)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    # start training
    _logger.info(f"Start training for {args.epochs} epochs")
    for epoch in range(start_epoch, args.epochs):
        if hasattr(loader_train, 'sampler'):
            loader_train.sampler.set_epoch(epoch)
        # one epoch training
        train_metrics = train_one_epoch(
                epoch, model, loader_train, optimizer, train_loss_fn, args,
                reg_loss_fn=reg_loss_fn,
                lr_scheduler=lr_scheduler, saver=saver, amp_autocast=amp_autocast,
                loss_scaler=loss_scaler, model_ema=model_ema, mixup_fn=mixup_fn, _logger=_logger)

        # distributed bn sync
        if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
            _logger.info("Distributing BatchNorm running means and vars")
            distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

        # calculate evaluation metric
        eval_metrics = validate(model, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, _logger=_logger)

        # model ema update
        if model_ema is not None and not args.model_ema_force_cpu:
            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')
            ema_eval_metrics = validate(model_ema.module, loader_eval, validate_loss_fn, args, amp_autocast=amp_autocast, log_suffix=' (EMA)', _logger=_logger)
            eval_metrics = ema_eval_metrics

        # lr_scheduler update
        if lr_scheduler is not None:
            # step LR for next epoch
            lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

        # output summary.csv
        if output_dir is not None:
            update_summary(
                epoch, train_metrics, eval_metrics, os.path.join(output_dir, 'summary.csv'),
                write_header=best_metric is None)

        # save checkpoint, print best metric
        if saver is not None:
            best_metric, best_epoch = saver.save_checkpoint(epoch, eval_metrics[eval_metric])
        torch.distributed.barrier()
    if best_metric is not None:
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

def train_one_epoch(
        epoch, model, loader, optimizer, loss_fn, args,
        reg_loss_fn=None,
        lr_scheduler=None, saver=None, amp_autocast=None,
        loss_scaler=None, model_ema=None, mixup_fn=None, _logger=None):
    # mixup setting
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    # statistical variables
    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    losses_m = AverageMeter()
    num_epochs = args.epochs + args.cooldown_epochs

    # model.train()
    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    num_updates = epoch * len(loader)
    
    att_step = args.attack_step * min(epoch, 5)/5
    att_eps=args.attack_eps
    att_it=args.attack_it

    alpha=0

    optimizer.zero_grad()
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_idx

        # processing input and target
        input, target = input.cuda(non_blocking=True), target.cuda(non_blocking=True)
        if mixup_fn is not None:
            input, target = mixup_fn(input, target)
        if args.channels_last:
            input=input.contiguous(memory_format=torch.channels_last)
        
        data_time_m.update(time.time() - end)

        # generate adv input
        if args.advtrain:
            input_advtrain = adv_generator(args, input, target, model, att_eps, att_it, att_step, random_start=False, attack_criterion=args.attack_criterion)

        # generate advprop input
        if args.advprop:
            model.apply(lambda m: setattr(m, 'bn_mode', 'adv'))
            input_advprop = adv_generator(args, input, target, model, 1/255, 1, 1/255, random_start=True, attack_criterion=args.attack_criterion, use_best=False)

        with amp_autocast():
            if args.advprop:
                outputs = model(input_advprop)
                adv_loss = loss_fn(outputs, target)
                model.apply(lambda m: setattr(m, 'bn_mode', 'clean'))
                outputs = model(input)
                loss = loss_fn(outputs, target) + adv_loss
            elif args.advtrain:
                output = model(input_advtrain)
                loss = loss_fn(output, target)
                loss_v = [loss]
            elif args.gradnorm:
                
                if args.random_start:
                    with torch.no_grad():
                        input = random_uniform_generator(input, args.mean, args.std, att_eps*args.random_start_eps_factor)

                input.requires_grad_(True)
                output = model(input)
                ce_loss = loss_fn(output, target)
                loss = args.ce_weight * ce_loss
                loss_v = [ce_loss]

                gradient = torch.autograd.grad(ce_loss, input, create_graph=True, retain_graph=True)[0]
                loss_reg = reg_loss_fn(gradient, input)

                alpha = max(0., min(args.alpha[0] + ((epoch-args.alpha_start_epoch) + (batch_idx // args.grad_accum) / (len(loader) / args.grad_accum)) * args.alpha[1], args.alpha[2]))
                if not (args.loss_fn == 'DBPTangent' or args.loss_fn == 'DBPChange' or args.loss_fn == 'DBPSparsity' or args.loss_fn == 'DBPAMHM'):
                    loss += args.gradnorm_weight * alpha * loss_reg
                    loss_v += [loss_reg]
                elif args.loss_fn == 'DBPSparsity':
                    norm_term, sparsity_term = loss_reg
                    loss += args.gradnorm_weight * alpha * (norm_term + sparsity_term)
                    loss_v += [norm_term, sparsity_term]
                else:
                    loss_reg_d0, loss_reg_d1 = loss_reg
                    loss += args.gradnorm_weight * alpha * (loss_reg_d0 + loss_reg_d1)
                    loss_v += [loss_reg_d0, loss_reg_d1]
            else:
                output = model(input)
                loss = loss_fn(output, target)
                loss_v = [loss]
            loss_v = [loss] + loss_v
            loss_v = torch.stack(loss_v, dim=0)

        if not args.distributed:
            losses_m.update(loss.item(), input.size(0))
        else:
            torch.cuda.synchronize()
            reduced_loss = reduce_tensor(loss_v.data, args.world_size)
            losses_m.update(reduced_loss.detach().cpu().numpy(), input.size(0))

        # optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss, optimizer,
                clip_grad=args.clip_grad, clip_mode=args.clip_mode,
                parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                create_graph=second_order)
        else:
            if args.grad_accum == 1:
                optimizer.zero_grad()
            loss.backward(create_graph=second_order)
            if (batch_idx + 1) % args.grad_accum == 0:
                if args.clip_grad is not None:
                    dispatch_clip_grad(
                        model_parameters(model, exclude_head='agc' in args.clip_mode),
                        value=args.clip_grad, mode=args.clip_mode)
                optimizer.step()
                optimizer.zero_grad()
        
        if model_ema is not None:
            if (batch_idx + 1) % args.grad_accum == 0:
                model_ema.update(model)
        
        if (batch_idx + 1) % args.grad_accum == 0:
            torch.cuda.synchronize()
            num_updates += 1
            batch_time_m.update(time.time() - end)

            # Save snapshot
            accum_batch_idx = batch_idx // args.grad_accum
            if (saver is not None) and (args.save_snapshot_for_inference is not None):
                if f'{epoch};{accum_batch_idx}' in args.save_snapshot_for_inference:
                    save_snapshot_for_inference(saver, epoch, accum_batch_idx)
                if (f'{epoch};-1' in args.save_snapshot_for_inference) and ((accum_batch_idx + 1) == (len(loader) // args.grad_accum)):
                    save_snapshot_for_inference(saver, epoch, accum_batch_idx)
                elif ((args.model == 'resnet50') or (args.model == 'resnet18')) and ((accum_batch_idx + 1) == (len(loader) // args.grad_accum)):
                    save_snapshot_for_inference(saver, epoch, accum_batch_idx)

            # # Collect gradient statistics
            # if (args.collect_gradient_statistics is not None) and (epoch == 0 or epoch == 1):
            #     save_gradient_statistics(args, model, epoch, accum_batch_idx)
            #     torch.distributed.barrier()
            

            if last_batch or (batch_idx // args.grad_accum) % args.log_interval == 0:
                lrl = [param_group['lr'] for param_group in optimizer.param_groups]
                lr = sum(lrl) / len(lrl)

                # model_ema.module.train()

                # if args.distributed:
                #     reduced_loss = reduce_tensor(loss_v.data, args.world_size)
                #     losses_m.update(reduced_loss.detach().cpu().numpy(), input.size(0))

                _logger.info(
                'Train: [{}/{}] [{:>4d}/{} ({:>3.0f}%)]  '
                'Loss: {loss_val} ({loss_avg})  '
                'Time: {batch_time.val:.3f}s, {rate:>7.2f}/s  '
                '({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                'LR: {lr:.3e}  '
                'Data: {data_time.val:.3f} ({data_time.avg:.3f})'.format(
                    epoch, num_epochs,
                    batch_idx // args.grad_accum, len(loader) // args.grad_accum,
                    100. * batch_idx / last_idx,
                    loss_val=formatted_array_str(losses_m.val, '#.4g'),
                    loss_avg=formatted_array_str(losses_m.avg, '#.4g'),
                    batch_time=batch_time_m,
                    rate=input.size(0) * args.world_size / batch_time_m.val,
                    rate_avg=input.size(0) * args.world_size / batch_time_m.avg,
                    lr=lr,
                    data_time=data_time_m))

            # # save checkpoint
            # if saver is not None and args.recovery_interval and (
            #         last_batch or (batch_idx + 1) % args.recovery_interval == 0):
            #     saver.save_recovery(epoch, batch_idx=batch_idx)

            # update lr scheduler
            if lr_scheduler is not None:
                lr_scheduler.step_update(num_updates=num_updates/(len(loader)//args.grad_accum), metric=losses_m.avg)

            end = time.time()
            # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    return OrderedDict([('loss', losses_m.avg)])

def validate(model, loader, loss_fn, args, amp_autocast=None, log_suffix='', _logger=None):
    batch_time_m = AverageMeter()
    losses_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    adv_losses_m = AverageMeter()
    adv_top1_m = AverageMeter()
    adv_top5_m = AverageMeter()


    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    for batch_idx, (input, target) in enumerate(loader):
        # read eval input
        last_batch = batch_idx == last_idx
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # normal eval process
        with torch.no_grad():
            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]
            
            loss = loss_fn(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = reduce_tensor(loss.data, args.world_size)
                acc1 = reduce_tensor(acc1, args.world_size)
                acc5 = reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            torch.cuda.synchronize()

            # record normal results
            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), output.size(0))
            top5_m.update(acc5.item(), output.size(0))

        # adv eval process
        if True:
            adv_input=adv_generator(args, input, target, model, 4/255, 10, 1/255, random_start=True, use_best=False, attack_criterion='regular')
            with torch.no_grad():
                with amp_autocast():
                    adv_output = model(adv_input)
                if isinstance(adv_output, (tuple, list)):
                    adv_output = adv_output[0]
                
                adv_loss = loss_fn(adv_output, target)
                adv_acc1, adv_acc5 = accuracy(adv_output, target, topk=(1, 5))

                if args.distributed:
                    adv_reduced_loss = reduce_tensor(adv_loss.data, args.world_size)
                    adv_acc1 = reduce_tensor(adv_acc1, args.world_size)
                    adv_acc5 = reduce_tensor(adv_acc5, args.world_size)
                else:
                    adv_reduced_loss = adv_loss.data

                torch.cuda.synchronize()

                # record adv results
                adv_losses_m.update(adv_reduced_loss.item(), adv_input.size(0))
                adv_top1_m.update(adv_acc1.item(), adv_output.size(0))
                adv_top5_m.update(adv_acc5.item(), adv_output.size(0))


        batch_time_m.update(time.time() - end)
        end = time.time()

        if last_batch or batch_idx % args.log_interval == 0:
            log_name = 'Test' + log_suffix
            _logger.info(
                '{0}: [{1:>4d}/{2}]  '
                'Time: {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f})  '
                'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})  '
                'AdvLoss: {adv_loss.val:>7.4f} ({adv_loss.avg:>6.4f})  '
                'AdvAcc@1: {adv_top1.val:>7.4f} ({adv_top1.avg:>7.4f})  '
                'AdvAcc@5: {adv_top5.val:>7.4f} ({adv_top5.avg:>7.4f})'.format(
                    log_name, batch_idx, last_idx, batch_time=batch_time_m,
                    loss=losses_m, top1=top1_m, top5=top5_m,
                    adv_loss=adv_losses_m, adv_top1=adv_top1_m, adv_top5=adv_top5_m))

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg), ('advloss', adv_losses_m.avg), ('advtop1', adv_top1_m.avg), ('advtop5', adv_top5_m.avg)])

    return metrics

def save_snapshot_for_inference(saver, epoch, iter):
    save_state = {
    'epoch': epoch,
    'arch': type(saver.model).__name__.lower(),
    'state_dict': get_state_dict(saver.model, saver.unwrap_fn),
    'optimizer': saver.optimizer.state_dict(),
    'version': 2,  # version < 2 increments epoch before save
    }
    if saver.args is not None:
        save_state['arch'] = saver.args.model
        save_state['args'] = saver.args
    if saver.amp_scaler is not None:
        save_state[saver.amp_scaler.state_dict_key] = saver.amp_scaler.state_dict()
    if saver.model_ema is not None:
        save_state['state_dict_ema'] = get_state_dict(saver.model_ema, saver.unwrap_fn)
    save_path = f'{saver.checkpoint_dir}/snapshots/snapshot-{epoch}-{iter}.pth.tar'
    torch.save(save_state, save_path)

def save_gradient_statistics(args, model, epoch, iter):
    meta_file='gradnorm_monitoring'
    imagenet_path=args.eval_dir
    dataloader_eval, dataset_eval = get_dataloader_for_visualization(args, root=imagenet_path, meta_file=meta_file, batch_size=args.batch_size)

    std_tensor=torch.Tensor(args.std).cuda(non_blocking=True)[None, :, None, None]
    mean_tensor=torch.Tensor(args.mean).cuda(non_blocking=True)[None, :, None, None]

    mode = model.training
    model.eval()
    gradient_norms = []
    for (input, target) in dataloader_eval:
        input = input.cuda()
        target = target.cuda()
        input = (input-mean_tensor)/std_tensor

        input.requires_grad_(True)
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)
        
        gradients = torch.autograd.grad(loss, input, create_graph=False, retain_graph=False)[0]
        gradients = args.batch_size*gradients.abs().sum((-3, -2, -1))
        gradient_norms.append(gradients.detach().cpu())
    gradient_norms = torch.concat(gradient_norms)
    with open(f'{args.output_dir}/gradnorms.txt', "a") as file:
        print(f'{epoch};{iter};{gradient_norms.numpy()}', file=file)

    if mode:
        model.train()

def random_uniform_generator(images, mean, std, eps):
    # denorm images to 0-1
    std_tensor=torch.Tensor(std).cuda(non_blocking=True)[None, :, None, None]
    mean_tensor=torch.Tensor(mean).cuda(non_blocking=True)[None, :, None, None]
    images=images*std_tensor+mean_tensor
    
    noise = torch.rand_like(images)
    noise.uniform_(-eps, eps)
    images = torch.clamp(images+noise, 0, 1)

    return (images - mean_tensor) / std_tensor


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Robust training script', parents=[get_args_parser()])
    args = parser.parse_args()
    opt = vars(args)
    if args.configs:
        opt.update(yaml.load(open(args.configs), Loader=yaml.FullLoader))
    
    args = argparse.Namespace(**opt)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    main(args, args_text)