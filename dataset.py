import numpy as np
import pdb
import torch
from torchvision import datasets
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
import torchvision


def get_dataset(name, path):
    if name.lower() == 'mvtec':
        return get_mvtec(path)


def get_mvtec(path):
    train_dir = os.path.join(path, 'mvtec/train')
    test_dir = os.path.join(path, 'mvtec/test')
    train_data = torchvision.datasets.ImageFolder(train_dir)
    test_data = torchvision.datasets.ImageFolder(test_dir)
    X_tr = np.array([np.asarray(datasets.folder.default_loader(s[0])) for s in train_data.samples])
    Y_tr = torch.from_numpy(np.array(train_data.targets))
    X_te = np.array([np.asarray(datasets.folder.default_loader(s[0])) for s in test_data.samples])
    Y_te = torch.from_numpy(np.array(test_data.targets))

    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name.lower() == 'mvtec':
        return DataHandler1
    else:
        return DataHandler2


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler2(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        return x, y, index

    def __len__(self):
        return len(self.X)


# handler for waal
def get_wa_handler(name):
    if name.lower() == 'mvtec':
        return  Wa_datahandler1


class Wa_datahandler1(Dataset):

    def __init__(self,X_1, Y_1, X_2, Y_2, transform = None):
        """
        :param X_1: covariate from the first distribution
        :param Y_1: label from the first distribution
        :param X_2:
        :param Y_2:
        :param transform:
        """
        self.X1 = X_1
        self.Y1 = Y_1
        self.X2 = X_2
        self.Y2 = Y_2
        self.transform = transform

    def __len__(self):

        # returning the minimum length of two data-sets

        return max(len(self.X1),len(self.X2))

    def __getitem__(self, index):
        Len1 = len(self.Y1)
        Len2 = len(self.Y2)

        # checking the index in the range or not

        if index < Len1:
            x_1 = self.X1[index]
            y_1 = self.Y1[index]

        else:

            # rescaling the index to the range of Len1
            re_index = index % Len1

            x_1 = self.X1[re_index]
            y_1 = self.Y1[re_index]

        # checking second datasets
        if index < Len2:

            x_2 = self.X2[index]
            y_2 = self.Y2[index]

        else:
            # rescaling the index to the range of Len2
            re_index = index % Len2

            x_2 = self.X2[re_index]
            y_2 = self.Y2[re_index]

        if self.transform is not None:

            x_1 = Image.fromarray(x_1)
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2)
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2