'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from torch.autograd import Variable


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class resnet_fea(nn.Module):
    def __init__(self, block, num_blocks):
        super(resnet_fea, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, img_size, intermediate=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out = F.avg_pool2d(out4, 4)
        out = out.view(out.size(0), -1)
        return out, [out1, out2, out3, out4]

class resnet_clf(nn.Module):
    def __init__(self, block, n_class=10):
        super(resnet_clf, self).__init__()
        self.linear = nn.Linear(128 * block.expansion * 4, n_class)

    def forward(self, x):
        # emb = x.view(x.size(0), -1)
        out = self.linear(x)
        return out, x

class resnet_dis(nn.Module):
    def __init__(self, embDim):
        super(resnet_dis, self).__init__()
        self.dis_fc1 = nn.Linear(embDim, 50)
        self.dis_fc2 = nn.Linear(50, 1)

    def forward(self, x):
        e1 = F.relu(self.dis_fc1(x))
        x = F.dropout(e1, training=self.training)
        x = self.dis_fc2(x)
        x  = torch.sigmoid(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, n_class=10, bayesian=False):
        super(ResNet, self).__init__()
        # self.in_planes = 16
        self.embDim = 128 * block.expansion * 4
        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(16)
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        # self.layer4 = self._make_layer(block, 128, num_blocks[3], stride=2)
        # self.linear = nn.Linear(128 * block.expansion, n_class)

        # self.dis_fc1 = nn.Linear(512, 50)
        # self.dis_fc2 = nn.Linear(50, 1)

        self.feature_extractor = resnet_fea(block, num_blocks)
        self.linear = resnet_clf(block, n_class)
        self.discriminator = resnet_dis(self.embDim)
        self.bayesian = bayesian

    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)

    # def feature_extractor(self, x): # feature extractor
    #     out = F.relu(self.bn1(self.conv1(x)))
    #     out = self.layer1(out)
    #     out = self.layer2(out)
    #     out = self.layer3(out)
    #     out = self.layer4(out)
    #     out = F.avg_pool2d(out, 4)
    #     emb = out.view(out.size(0), -1)
    #     return emb


    def forward(self, x, intermediate=False):
        out, in_values = self.feature_extractor(x, x.shape[2])
        # apply dropout to approximate the bayesian networks
        out = F.dropout(out, p=0.2, training=self.bayesian)
        # emb = emb.view(emb.size(0), -1)
        out, emb = self.linear(out)
        if intermediate == True:
            return out, emb, in_values
        else:
            return out, emb

    def get_embedding_dim(self):
        return self.embDim


def ResNet18_64(n_class, bayesian=False):
    return ResNet(BasicBlock, [2,2,2,2], n_class=n_class, bayesian=bayesian)

def ResNet34_64(n_class, bayesian=False):
    return ResNet(BasicBlock, [3,4,6,3], n_class=n_class, bayesian=bayesian)

def ResNet50_64(n_class, bayesian=False):
    return ResNet(Bottleneck, [3,4,6,3], n_class=n_class, bayesian=bayesian)

def ResNet101_64(n_class, bayesian=False):
    return ResNet(Bottleneck, [3,4,23,3], n_class=n_class, bayesian=bayesian)

def ResNet152_64(n_class, bayesian=False):
    return ResNet(Bottleneck, [3,8,36,3], n_class=n_class, bayesian=bayesian)

def test():
    net = ResNet18()
    y = net(Variable(torch.randn(1,3,32,32)))
    print(y.size())

# test()
