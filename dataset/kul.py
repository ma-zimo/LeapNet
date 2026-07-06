import torch

import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle

class readKUL:
    def __init__(self):
        self.train_data = np.load('/data4/zimo/KUL/train_data.npy')
        self.train_labels = np.load('/data4/zimo/KUL/train_labels.npy')
        self.test_data = np.load('/data4/zimo/KUL/test_data.npy')
        self.test_labels = np.load('/data4/zimo/KUL/test_labels.npy')


class KUL(Dataset):
    def __init__(self, mode):
        self.data = torch.from_numpy(getattr(readKUL(), '{}_data'.format(mode))).float().permute(0, 3, 1, 2)
        self.target = torch.from_numpy(getattr(readKUL(), '{}_labels'.format(mode))).long()

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


def load_kul(args):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(KUL('train'),
                                               batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(KUL('test'),
                                              batch_size=args.batch_size_validation, shuffle=False, **kwargs)

    return train_loader, test_loader
