import os
import torch

import os.path
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets.utils import check_integrity, download_url, verify_str_arg
from torchvision.datasets.vision import VisionDataset

DATA_DESC = {
    'data': 'tiny-imagenet',
    'classes': tuple(range(0, 200)),
    'num_classes': 200,
    'mean': [0.4802, 0.4481, 0.3975],
    'std': [0.2302, 0.2265, 0.2262],
}


class TinyImagenet(VisionDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", tuple(self.split_list.keys()))
        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        loaded_npz = np.load(os.path.join(self.root, self.filename))
        self.data = loaded_npz['image']
        self.targets = loaded_npz["label"].tolist()
        print(split + ' images size:', self.data.shape)
        print(split + ' labels size:', len(self.targets))

    split_list = {
        "train": [
            "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/dataset/tiny-imagenet-200/train.npz",
            "train.npz",
            "db414016436353892fdf00cb30b9ee57",
        ],
        "val": [
            "https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/others/dataset/tiny-imagenet-200/val.npz",
            "val.npz",
            "7762694b6217fec8ba1a7be3c20ef218",
        ],
    }

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    def _check_integrity(self) -> bool:
        fpath = os.path.join(self.root, self.filename)
        print(fpath)
        return check_integrity(fpath, self.file_md5)

    def download(self) -> None:
        download_url(self.url, self.root, self.filename, self.file_md5)

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)


def load_tinyimagenet(args):
    """
    Returns Tiny Imagenet-200 train and test dataloaders.
    Arguments:
        args: should contain args.data_dir, args.batch_size, args.batch_size_validation, and args.use_augmentation
    Returns:
        train_loader, test_loader
    """
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    if args.augment == 'base':
        train_transform = transforms.Compose([
            transforms.RandomCrop(64, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = test_transform

    train_dataset = TinyImagenet(root='/data1/zimo/tiny-imagenet', split='train',
                                 download=True, transform=train_transform)
    test_dataset = TinyImagenet(root='/data1/zimo/tiny-imagenet', split='val',
                                download=True, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size_validation, shuffle=False, **kwargs)

    return train_loader, test_loader

