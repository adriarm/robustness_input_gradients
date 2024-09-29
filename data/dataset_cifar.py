import torch
import os
from timm.data import Mixup, AugMixDataset, create_transform
from timm.data.distributed_sampler import OrderedDistributedSampler, RepeatAugSampler
from torchvision import datasets
from PIL import Image
import torchvision

class ImageNet(torch.utils.data.Dataset):
    """move to tools data_set"""
    def __init__(self, root, meta_file='./src_data/val.txt', transform=None): # ./imagenet_val_1k.txt

        self.data_dir = root
        self.meta_file = meta_file
        self.transform = transform
        self._indices = []

        for line in open(meta_file, encoding="utf-8"):
            temp_names=line.strip().split(' ')
            img_path, label = temp_names[0], temp_names[1]
            self._indices.append((os.path.join(self.data_dir, img_path), label))

    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index]
        img = Image.open(img_path).convert('RGB')
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

def build_dataset(args, num_aug_splits=0):
    # build dataset
    #dataset_train = datasets.ImageFolder(root=args.train_dir, transform=None)
    #dataset_eval = datasets.ImageFolder(root=args.eval_dir, transform=None)
    dataset_train = torchvision.datasets.CIFAR10(root='/data/vision/torralba/naturally_robust_models/input_norm/cifar10_dataset', train=True,
                                        download=False, transform=None)
    dataset_eval = torchvision.datasets.CIFAR10(root='/data/vision/torralba/naturally_robust_models/input_norm/cifar10_dataset', train=False,
                                       download=False, transform=None)
    # dataset_eval=ImageNet(root=args.eval_dir)

    # wrap dataset_train in AugMix helper
    if num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # build transform
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = args.interpolation
    re_num_splits = 0
    if args.resplit:
        # apply RE to second half of batch if no aug split otherwise line up with aug split
        re_num_splits = num_aug_splits or 2
    dataset_train.transform = create_transform(
        args.input_size,
        is_training=True,
        use_prefetcher=False,
        no_aug=args.no_aug,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        interpolation=train_interpolation,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct,
        tf_preprocessing=False,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_num_splits=re_num_splits,
        separate=num_aug_splits > 0
    )
    #dataset_train.transform = torchvision.transforms.Compose([dataset_train.transform, torchvision.transforms.Grayscale(3)])

    dataset_eval.transform = create_transform(
        args.input_size,
        is_training=False,
        use_prefetcher=False,
        interpolation=args.interpolation,
        mean=args.mean,
        std=args.std,
        crop_pct=args.crop_pct
    )
    #dataset_eval.transform = torchvision.transforms.Compose([dataset_eval.transform, torchvision.transforms.Grayscale(3)])

    # create sampler
    sampler_train = None
    sampler_eval = None
    if args.distributed and not isinstance(dataset_train, torch.utils.data.IterableDataset):
        if args.aug_repeats:
            sampler_train = RepeatAugSampler(dataset_train, num_repeats=args.aug_repeats)
        else:
            sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    else:
        assert args.aug_repeats == 0, "RepeatAugment not currently supported in non-distributed or IterableDataset use"
    sampler_eval = OrderedDistributedSampler(dataset_eval)

    # create dataloader
    dataloader_train = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_train,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=True
    )
    dataloader_eval = torch.utils.data.DataLoader(
        dataset=dataset_eval,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=sampler_eval,
        collate_fn=None,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes)
        mixup_fn = Mixup(**mixup_args)

    return dataloader_train, dataloader_eval, mixup_fn