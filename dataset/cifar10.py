import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle

def load_cifar10(args):
    """
    Returns CIFAR10 train and test dataloaders.
    Arguments:
        args: should contain args.data_dir, args.batch_size, args.batch_size_validation, and args.use_augmentation
    Returns:
        train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}

    test_transform = transforms.Compose([transforms.ToTensor()])

    if args.augment == 'base':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor()
        ])
    else:
        train_transform = test_transform

    train_dataset = torchvision.datasets.CIFAR10(root='/data3/zimo/cifar10', train=True, download=True,
                                                 transform=train_transform)
    test_dataset = torchvision.datasets.CIFAR10(root='/data3/zimo/cifar10', train=False, download=True,
                                                transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_validation, shuffle=False,
                                              **kwargs)

    return train_loader, test_loader