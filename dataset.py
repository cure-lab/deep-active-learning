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
    if name.lower() == 'mnist':
        return get_MNIST(path)
    elif name.lower() == 'fashionmnist':
        return get_FashionMNIST(path)
    elif name.lower() == 'svhn':
        return get_SVHN(path)
    elif name.lower() == 'cifar10':
        return get_CIFAR10(path)
    elif name.lower() == 'gtsrb':
        return get_GTSRB(path)
    elif name.lower() == 'tinyimagenet':
        return get_tinyImageNet(path)


def get_tinyImageNet(path):
    # 100000 train 10000 test
    raw_tr = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/train')
    raw_te = datasets.ImageFolder(path + '/tinyImageNet/tiny-imagenet-200/val')
    f = open('dataset/tinyImageNet/tiny-imagenet-200/val/val_annotations.txt')
    val_dict = {}
    for line in f.readlines():
        val_dict[line.split()[0]] = raw_tr.class_to_idx[line.split()[1]]

    X_tr,Y_tr,X_te, Y_te = [],[],[],[]
    count=0
    coun_list = [1000*(x+1) for x in range(100)]
    
    for ct in coun_list:
        while count < ct:
            image,target = raw_tr[count]
            X_tr.append(np.array(image))
            Y_tr.append(target)
            count += 1
    count=0
    coun_list = [1000*(x+1) for x in range(10)]
    for ct in coun_list:
        while count < ct:
            image,target = raw_te[count]
            img_pth = raw_te.imgs[count][0].split('/')[-1]
            X_te.append(np.array(image))
            Y_te.append(val_dict[img_pth])
            count += 1
    return torch.from_numpy(np.array(X_tr)), torch.from_numpy(np.array(Y_tr)), torch.from_numpy(np.array(X_te)), torch.from_numpy(np.array(Y_te))

def get_MNIST(path):
    raw_tr = datasets.MNIST(path + '/mnist', train=True, download=True)
    raw_te = datasets.MNIST(path + '/mnist', train=False, download=True)
    print(type(raw_tr))
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_FashionMNIST(path):
    raw_tr = datasets.FashionMNIST(path + '/fashionmnist', train=True, download=True)
    raw_te = datasets.FashionMNIST(path + '/fashionmnist', train=False, download=True)
    X_tr = raw_tr.data
    Y_tr = raw_tr.targets
    X_te = raw_te.data
    Y_te = raw_te.targets
    return X_tr, Y_tr, X_te, Y_te

def get_SVHN(path):
    data_tr = datasets.SVHN(path, split='train', download=True)
    data_te = datasets.SVHN(path, split='test', download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(data_tr.labels)
    X_te = data_te.data
    Y_te = torch.from_numpy(data_te.labels)
    return X_tr, Y_tr, X_te, Y_te

def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path + '/cifar10', train=True, download=True)
    data_te = datasets.CIFAR10(path + '/cifar10', train=False, download=True)
    X_tr = data_tr.data
    # print(np.array(X_tr[0]).shape)
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))
    return X_tr, Y_tr, X_te, Y_te

def get_GTSRB(path):
    train_dir = os.path.join(path, 'gtsrb/train')
    test_dir = os.path.join(path, 'gtsrb/test')
    train_data = torchvision.datasets.ImageFolder(train_dir)
    test_data = torchvision.datasets.ImageFolder(test_dir)
    X_tr = np.array([np.asarray(datasets.folder.default_loader(s[0])) for s in train_data.samples])
    Y_tr = torch.from_numpy(np.array(train_data.targets))
    X_te = np.array([np.asarray(datasets.folder.default_loader(s[0])) for s in test_data.samples])
    Y_te = torch.from_numpy(np.array(test_data.targets))

    return X_tr, Y_tr, X_te, Y_te


def get_handler(name):
    if name.lower() == 'mnist':
        return DataHandler1
    elif name.lower() == 'fashionmnist':
        return DataHandler1
    elif name.lower() == 'svhn':
        return DataHandler2
    elif name.lower() == 'cifar10':
        return DataHandler3
    elif name.lower() == 'gtsrb':
        return DataHandler3
    elif name.lower() == 'tinyimagenet':
        return DataHandler3
    else:
        return DataHandler4


class DataHandler1(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = x.numpy() if not isinstance(x, np.ndarray) else x
            x = Image.fromarray(x, mode='L')
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
        if self.transform is not None:
            x = Image.fromarray(np.transpose(x, (1, 2, 0)))
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)

class DataHandler3(Dataset):
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

class DataHandler4(Dataset):
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
    if name.lower() == 'fashionmnist':
        return  Wa_datahandler1
    elif name.lower() == 'svhn':
        return Wa_datahandler2
    elif name.lower() == 'cifar10':
        return  Wa_datahandler3
    elif name.lower() == 'mnist':
        return Wa_datahandler1
    elif name.lower() == 'gtsrb':
        return Wa_datahandler3


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
            # print (x_1)
            x_1 = Image.fromarray(x_1, mode='L')
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(x_2, mode='L')
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2



class Wa_datahandler2(Dataset):

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

            x_1 = Image.fromarray(np.transpose(x_1, (1, 2, 0)))
            x_1 = self.transform(x_1)

            x_2 = Image.fromarray(np.transpose(x_2, (1, 2, 0)))
            x_2 = self.transform(x_2)

        return index,x_1,y_1,x_2,y_2


class Wa_datahandler3(Dataset):

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

# get_CIFAR10('./dataset')
