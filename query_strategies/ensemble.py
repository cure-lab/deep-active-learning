import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from .strategy import Strategy
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

def kl_div_with_logit(q_logit, p_logit):

    q = F.softmax(q_logit, dim=1)
    logq = F.log_softmax(q_logit, dim=1)
    logp = F.log_softmax(p_logit, dim=1)

    qlogq = ( q *logq).sum(dim=1).mean(dim=0)
    qlogp = ( q *logp).sum(dim=1).mean(dim=0)

    return qlogq - qlogp


def _l2_normalize(d):

    d = d.numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2, 3))).reshape((-1, 1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)


def vat_loss(model, ul_x, ul_y, xi=1e-6, eps=2.5, num_iters=1):

    # find r_adv

    d = torch.Tensor(ul_x.size()).normal_()
    for i in range(num_iters):
        d = xi *_l2_normalize(d)
        # print('d:',d.shape)
        d = Variable(d.cuda(), requires_grad=True)
        y_hat = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


class Ensemble(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(Ensemble, self).__init__(X, Y, idxs_lb, net, handler, args)

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def ensemble_mean_var(self, x):
        ensemble = self.clf
        en_mean = 0
        en_var = 0

        for model in ensemble:
            mean, var = model(x)
            en_mean += mean
            en_var += var + mean ** 2

        en_mean /= len(ensemble)
        en_var /= len(ensemble)
        en_var -= en_mean ** 2
        return en_mean, en_var

    # 我这里是想把预测结果全部加在一起，直接求哪一行的数字最大，那么那一行就是整体预测的分类结果
    # 但是我不确定需不需要给model输出结果套一层softmax，因为model最后一层是一个batch norm
    def ensemble_predict(self,x):

        f = True
        for model in self.clf:
            y_pred = model(x)
            if f:
                sum_y = y_pred
                f = False
            else:
                sum_y += y_pred
        return sum_y

    def ensemble_train(self, epoch, labeled_data, unlabeled_data, optimizer):
        for model in self.clf:
            model.train()
        accFinal = 0.
        data_len = len(labeled_data[0].dataset.X)
        labeled_data = [self.read_data(loader_tr) for loader_tr in labeled_data]
        # data 差一层循环
        batch_len = int(data_len/self.args['loader_tr_args']['batch_size'])
        for idx in range(batch_len):
            for model_idx, model in enumerate(self.clf):
                x, y = next(labeled_data[model_idx])
                x, y = Variable(x.cuda()), Variable(y.cuda())
                # print('y_shape',y.shape)
                unlabeled_x = next(unlabeled_data[model_idx])
                unlabeled_x = Variable(unlabeled_x.cuda())
                nll = nn.NLLLoss(reduction='mean')
                y_pred= model(x)
                nll_loss = nll(y_pred, y)
                # print(unlabeled_x.shape)
                unlabeled_y = model(unlabeled_x)
                # print('un y_pred', unlabeled_y.shape)
                v_loss = vat_loss(model, unlabeled_x, unlabeled_y, eps=2.5)
                loss = v_loss + nll_loss
                # if 'vatent':
                #     loss += entropy_loss(unlabeled_y)

                optimizer[model_idx].zero_grad()
                loss.backward()
                accFinal += torch.sum((torch.max(y_pred, 1)[1] == y).float()).data.item()
                # clamp gradients, just in case
                for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
                optimizer[model_idx].step()
        return accFinal / (data_len * len(self.clf))

    def train(self):


        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args['n_epoch']
        self.clf = [net.apply(weight_reset).cuda() for net in self.net]
        optimizer = [optim.Adam(model.parameters(), lr=self.args['lr'], weight_decay=0) for model in self.clf]

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        labeled_data = [DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=self.args['transform']), shuffle=True,
                               **self.args['loader_tr_args']) for i in range(len(self.clf))]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_data = [self.read_data(DataLoader(
            self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                         transform=self.args['transform']), shuffle=True,
            **self.args['loader_tr_args']), labels=False) for i in range(len(self.clf))]

        epoch = 1
        accCurrent = 0.
        while accCurrent < 0.99:
            accCurrent = self.ensemble_train(epoch, labeled_data, unlabeled_data, optimizer)
            epoch += 1
            print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            if (epoch % 50 == 0) and (accCurrent < 0.2): # reset if not converging
                new_clf = []
                for model in self.clf:
                    model = self.net.apply(weight_reset)
                    new_clf.append(model)
                    optimizer = optim.Adam(model.parameters(), lr = self.args['lr'], weight_decay=0)
                self.clf = new_clf

    def predict(self, X, Y):
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        else:
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        for model in self.clf:
            model.eval()

        P = torch.zeros(len(Y)).long()

        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                y_pred = self.ensemble_predict(x)
                pred = y_pred.max(1)[1]
                P[idxs]  = pred.data.cpu()
        return P

    # 首先求预测结果，然后计算所有模型在最终预测结果那一行的方差
    # 比如模型结果是第二类，那么就计算【五个模型在预测值的第二行】的方差作为模型预测的不一致
    # 这个原论文没有，学姐要是有更好的可以改掉
    # 学姐记得检查一下，这个我不知道会不会写错了
    def ensemble_predict_var(self,x):
        pred_list = []
        y_pred = self.ensemble_predict(x)
        pred = y_pred.max(1)[1].data.cpu()
        for model in self.clf:
            y_pred = model(x)
            pred_list.append(y_pred[:,pred])

        return np.var(pred_list)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_loader = DataLoader(
            self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                         transform=self.args['transform']), shuffle=True,
            **self.args['loader_tr_args'])

        for model in self.clf:
            model.eval()
        P = []
        with torch.no_grad():
            for x, y, idxs in unlabeled_loader:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                var = self.ensemble_predict_var(x)
                P.append([var,idxs])
        P.sort(key = lambda x : x[0])
        raw_idx = []
        for idx, tup in enumerate(P):
            if idx < n:
                raw_idx.append(tup[1])
        return idxs_unlabeled[raw_idx]