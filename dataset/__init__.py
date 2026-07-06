import os
import torch

from .cifar10 import load_cifar10
# from .cifar100 import load_cifar100
# from .svhn import load_svhn
from .kul import load_kul
from .tiny_imagenet import load_tinyimagenet
from .gtsrb import load_gtsrb


def load_dataset(args):
    # dataset = os.path.basename(os.path.normpath(data_dir))
    if args.dataset == 'gtsrb':
        train_dataloader, test_dataloader = load_gtsrb(args)
        return train_dataloader, test_dataloader
    elif args.dataset == 'kul':
        train_dataloader, test_dataloader = load_kul(args)
        return train_dataloader, test_dataloader
    elif args.dataset == 'tiny_imagenet':
        train_dataloader, test_dataloader = load_tinyimagenet(args)
        return train_dataloader, test_dataloader
    elif args.dataset == 'cifar10':
        train_dataloader, test_dataloader = load_cifar10(args)
        return train_dataloader, test_dataloader
    else:
        print("dataset no available")