'''Active Learning Procedure in PyTorch.
Reference:
[Yoo et al. 2019] Learning Loss for Active Learning (https://arxiv.org/abs/1905.03677)
'''

from torch.cuda import device
import torch.nn.functional as F
import numpy as np
from .strategy import Strategy



# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
# import torchvision.models as models
from torchvision.datasets import CIFAR10


class SubsetSequentialSampler(torch.utils.data.Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.
    Arguments:
        indices (sequence): a sequence of indices
    """
    def __init__(self, indices):
        super().__init__(indices)
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


MARGIN = 1.0 # xi
WEIGHT = 1.0 # lambda


LR = 0.1
MILESTONES = [160]
EPOCHL = 120 # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model

MOMENTUM = 0.9
WDECAY = 5e-4

# 源代码写的只有四层，因此只能对应四层的目标网络，并且目标网络需要输出每层的结果作为lossnet的输入
class LossNet(nn.Module):
    def __init__(self, feature_sizes=[32, 16, 8, 4], num_channels=[16, 32, 64, 128], interm_dim=128):
        super(LossNet, self).__init__()

        self.GAP1 = nn.AvgPool2d(feature_sizes[0])
        self.GAP2 = nn.AvgPool2d(feature_sizes[1])
        self.GAP3 = nn.AvgPool2d(feature_sizes[2])
        self.GAP4 = nn.AvgPool2d(feature_sizes[3])

        self.FC1 = nn.Linear(num_channels[0], interm_dim)
        self.FC2 = nn.Linear(num_channels[1], interm_dim)
        self.FC3 = nn.Linear(num_channels[2], interm_dim)
        self.FC4 = nn.Linear(num_channels[3], interm_dim)

        self.linear = nn.Linear(4 * interm_dim, 1)

    def forward(self, features):
        out1 = self.GAP1(features[0])
        out1 = out1.view(out1.size(0), -1)
        out1 = F.relu(self.FC1(out1))

        out2 = self.GAP2(features[1])
        out2 = out2.view(out2.size(0), -1)
        out2 = F.relu(self.FC2(out2))

        out3 = self.GAP3(features[2])
        out3 = out3.view(out3.size(0), -1)
        out3 = F.relu(self.FC3(out3))

        out4 = self.GAP4(features[3])
        out4 = out4.view(out4.size(0), -1)
        out4 = F.relu(self.FC4(out4))

        out = self.linear(torch.cat((out1, out2, out3, out4), 1))
        return out


def LossPredLoss(input, target, margin=1.0, reduction='mean'):
    assert len(input) % 2 == 0, 'the batch size is not even.'
    assert input.shape == input.flip(0).shape

    input = (input - input.flip(0))[
            :len(input) // 2]  # [l_1 - l_2B, l_2 - l_2B-1, ... , l_B - l_B+1], where batch_size = 2B
    target = (target - target.flip(0))[:len(target) // 2]
    target = target.detach()

    one = 2 * torch.sign(torch.clamp(target, min=0)) - 1  # 1 operation which is defined by the authors

    if reduction == 'mean':
        loss = torch.sum(torch.clamp(margin - one * input, min=0))
        loss = loss / input.size(0)  # Note that the size of input is already halved
    elif reduction == 'none':
        loss = torch.clamp(margin - one * input, min=0)
    else:
        NotImplementedError()
        return

    return loss


class LearningLoss4AL(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(LearningLoss4AL, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.loss_module = LossNet().cuda()


    def train_epoch(self, epoch, loader_tr, optimizers,criterion):
        self.clf.train()
        self.loss_module.train()
        accFinal = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device)
            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()

            scores, _, features = self.clf(x, intermediate=True)
            # print (len(scores), len(features))
            # print ("output shape", scores[0].shape, scores[1].shape)
            target_loss = criterion(scores, y)

            if epoch > 120:
                # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
            pred_loss = self.loss_module(features)
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss 	= LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss			= m_backbone_loss + WEIGHT * m_module_loss


            loss.backward()
            optimizers['backbone'].step()
            optimizers['module'].step()



            accFinal += torch.sum((torch.max(scores,1)[1] == y).float()).data.item()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
        return accFinal / len(loader_tr.dataset.X)

    def train(self, n_epoch, alpha=0.):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        # n_epoch = self.args['n_epoch']
        self.clf = self.net.apply(weight_reset).cuda()
        criterion = nn.CrossEntropyLoss(reduction='none')
        optim_backbone = optim.SGD(self.clf.parameters(), lr=LR,
                                   momentum=MOMENTUM, weight_decay=WDECAY)
        optim_module = optim.SGD(self.loss_module.parameters(), lr=LR,
                                 momentum=MOMENTUM, weight_decay=WDECAY)
        sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
        sched_module = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

        optimizers = {'backbone': optim_backbone, 'module': optim_module}
        schedulers = {'backbone': sched_backbone, 'module': sched_module}
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        transform = self.args['transform_tr'] if not self.pretrained else None
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=transform), shuffle=True,
                               **self.args['loader_tr_args'])

        epoch = 0
        accCurrent = 0.
        accOld = 0.
        while epoch < n_epoch:
			# epoch += 1
            schedulers['backbone'].step()
            schedulers['module'].step()
            accCurrent = self.train_epoch(epoch, loader_tr, optimizers, criterion)
            epoch += 1
            print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            if abs(accCurrent - accOld) < 0.0002:
                break
            else:
                accOld = accCurrent

    def get_uncertainty(self,models, unlabeled_loader):
        models['backbone'].eval()
        models['module'].eval()
        uncertainty = torch.tensor([]).cuda()

        with torch.no_grad():
            for (inputs, labels,idx) in unlabeled_loader:
                inputs = inputs.cuda()
                # labels = labels.cuda()
                scores, _, features = models['backbone'](inputs, intermediate= True)
                pred_loss = models['module'](features)  # pred_loss = criterion(scores, labels) # ground truth loss
                pred_loss = pred_loss.view(pred_loss.size(0))

                uncertainty = torch.cat((uncertainty, pred_loss), 0)

        return uncertainty.cpu()

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_loader = DataLoader(
            self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                         transform=self.args['transform_te']), shuffle=True,
            **self.args['loader_tr_args'])
        models = {'backbone': self.clf, 'module': self.loss_module}
        uncertainty = self.get_uncertainty(models, unlabeled_loader)

        # Index in ascending order
        arg_sort = np.argsort(uncertainty)
        return idxs_unlabeled[arg_sort[:n]]