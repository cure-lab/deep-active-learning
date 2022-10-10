# linear model and mlp class
from torch import nn
import numpy as np
import torch.nn.functional as F
import torch


class linMod(nn.Module):
    def __init__(self, dim, n_class):
        super(linMod, self).__init__()
        self.dim = dim
        self.lm = nn.Linear(int(np.prod(dim)), n_class)
    def forward(self, x):
        x = x.view(-1, int(np.prod(self.dim)))
        out = self.lm(x)
        return out, x
    def get_embedding_dim(self):
        return int(np.prod(self.dim))

# mlp model class
class mlpMod(nn.Module):
    def __init__(self, dim, n_class, embSize=256):
        super(mlpMod, self).__init__()
        self.embSize = embSize
        self.dim = int(np.prod(dim))
        self.lm1 = nn.Linear(self.dim, embSize)
        self.lm2 = nn.Linear(embSize, n_class)
    def forward(self, x):
        x = x.view(-1, self.dim)
        emb = F.relu(self.lm1(x))
        out = self.lm2(emb)
        return out, emb
    def get_embedding_dim(self):
        return self.embSize


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

        x1 = F.relu(F.max_pool2d(self.conv1(x), 2))
        x2 = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x1)), 2))
        # print (x.shape)
        x = x2.view(x2.shape[0], 320)

        return  x, [x1, x2]

class Net1_clf(nn.Module):
    """
    Classifier network, also give the latent space and embedding dimension
    """
    def __init__(self, n_class):
        super(Net1_clf,self).__init__()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, n_class)

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

# net
class LeNet(nn.Module):
    def __init__(self, n_class=10, bayesian=False):
        super(LeNet, self).__init__()
        self.feature_extractor = Net1_fea()
        self.linear = Net1_clf(n_class)
        self.discriminator = Net1_dis()
        self.bayesian = bayesian
    
    def forward(self, x, intermediate=False):
        x, in_values = self.feature_extractor(x)
        x = F.dropout(x, p=0.2, training=self.bayesian)
        x, e1 = self.linear(x)
        
        if intermediate == True:
            return x, e1, in_values
        else:
            return x, e1

    def get_embedding_dim(self):
        return 50
