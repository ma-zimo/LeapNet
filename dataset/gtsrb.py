import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.dataset import Dataset
import pickle

def load_batch(fpath):
    images = []
    labels = []
    num_classes = 43
    with open(fpath, 'rb') as rfile:
        train_dataset =  pickle.load(rfile)
    for image in train_dataset['features']:
        # print(image.min(),image.max())
        images.append((image/255)-.5)
    for label in train_dataset['labels']:
        # labels.append(np.eye(num_classes)[label])
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)

    return images, labels
class readGTSRB:
    def __init__(self):
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.validation_data = []
        self.validation_labels = []

        # load train data

        img, lab = load_batch('/data4/zimo/gtsrb/train.p')
        self.train_data.extend(img)
        self.train_labels.extend(lab)

        self.train_data = np.array(self.train_data, dtype=np.float32)
        self.train_labels = np.array(self.train_labels)

        # load test data

        img, lab = load_batch('/data4/zimo/gtsrb/test.p')
        self.test_data.extend(img)
        self.test_labels.extend(lab)

        self.test_data = np.array(self.test_data, dtype=np.float32)
        self.test_labels = np.array(self.test_labels)

        # load validation data

        img, lab = load_batch('/data4/zimo/gtsrb/valid.p')
        self.validation_data.extend(img)
        self.validation_labels.extend(lab)

        self.validation_data = np.array(self.validation_data, dtype=np.float32)
        self.validation_labels = np.array(self.validation_labels)


class GTSRB(Dataset):
    def __init__(self, mode):
        self.data = torch.from_numpy(getattr(readGTSRB(), '{}_data'.format(mode))).float().permute(0, 3, 1, 2)
        # print(self.data.shape)
        self.target = torch.from_numpy(getattr(readGTSRB(), '{}_labels'.format(mode))).long()
        # print(self.target[0])

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        return x, y

    def __len__(self):
        return len(self.data)


def load_gtsrb(args):
    kwargs = {'num_workers': 1, 'pin_memory': True, 'drop_last': False}
    train_loader = torch.utils.data.DataLoader(GTSRB('train'), batch_size=args.batch_size, shuffle=True, **kwargs)
    # train_loader = torch.utils.data.DataLoader(GTSRB('train'),batch_size=1, shuffle=True, **kwargs)

    # mean,std, minimum, maximum = utils.get_mean_and_std(GTSRB('train'))
    # print(mean,std, minimum, maximum)

    # valid_loader = torch.utils.data.DataLoader(GTSRB('validation'),
    # batch_size=1, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(GTSRB('test'),
                                              batch_size=args.batch_size_validation, shuffle=False, **kwargs)

    # return train_loader, valid_loader, test_loader
    return train_loader, test_loader

