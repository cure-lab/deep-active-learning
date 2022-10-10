# WAAL net prepared for MNIST and Fashion_mnist
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn.init as init

def get_wa_net(name):
    if name.lower() == 'fashionmnist':
        return Net1_fea, Net1_clf, Net1_dis
    # elif name.lower() == 'svhn':
    #     return VGG_10_fea, VGG_10_clf, VGG_10_dis
    elif name.lower() == 'cifar10':
        return  VGG_10_fea, VGG_10_clf, VGG_10_dis
    elif name.lower() == 'mnist': 
        return Net1_fea, Net1_clf, Net1_dis
    


# net_1  for Mnist and Fashion_mnist

class Net1_fea(nn.Module):
    """
    Feature extractor network
    """

    def __init__(self):
        super(Net1_fea, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

    def forward(self,x):

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print (x.shape)
        x = x.view(x.shape[0], 320)

        return  x

class Net1_clf(nn.Module):
    """
    Classifier network, also give the latent space and embedding dimension
    """

    def __init__(self):
        super(Net1_clf,self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self,x):

        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)

        return x, e1

    def get_embedding_dim(self):

        return 50

class Net1_dis(nn.Module):

    """
    Discriminator network, output with [0,1] (sigmoid function)
    """
    def __init__(self):
        super(Net1_dis,self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self,x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x




# VGG_three parts
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


## VGG for CIFAR 10/SVHN (since they are 32 * 32)

class VGG_10_fea(nn.Module):

    def __init__(self):

        super(VGG_10_fea, self).__init__()
        # the vgg model can be changed to vgg11/vgg16
        # vgg 11 for svhn
        # vgg 16 for cifar 10 and cifar 100
        self.features = self._make_layers(cfg['VGG16'])

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)

        return out

    def _make_layers(self, cfg):

        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_10_clf(nn.Module):

    def __init__(self):

        super(VGG_10_clf, self).__init__()
        self.fc1 = nn.Linear(512,50)
        self.fc2 = nn.Linear(50,10)

    def forward(self,x):
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)

        return x, e1

    def get_embedding_dim(self):

        return 50


class VGG_10_dis(nn.Module):

    def __init__(self):

        super(VGG_10_dis,self).__init__()
        self.fc1 = nn.Linear(512, 50)
        self.fc2 = nn.Linear(50, 1)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self,x):

        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x)

        return x


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)