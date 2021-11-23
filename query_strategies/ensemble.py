import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
from .strategy import Strategy
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import time
from copy import deepcopy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time
import warnings
warnings.filterwarnings("ignore")
# Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles
# Paper only proposes a method to measure data uncertainty, we apply it to active learning

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
        y_hat, e1 = model(ul_x + d)
        delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
        delta_kl.backward()

        d = d.grad.data.clone().cpu()
        model.zero_grad()

    d = _l2_normalize(d)
    d = Variable(d.cuda())
    r_adv = eps *d
    # compute lds
    y_hat, e1 = model(ul_x + r_adv.detach())
    delta_kl = kl_div_with_logit(ul_y.detach(), y_hat)
    return delta_kl


def entropy_loss(ul_y):
    p = F.softmax(ul_y, dim=1)
    return -(p*F.log_softmax(ul_y, dim=1)).sum(dim=1).mean(dim=0)


class ensemble(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ensemble, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    # def ensemble_mean_var(self, x):
    #     ensemble = self.clf
    #     en_mean = 0
    #     en_var = 0

    #     for model in ensemble:
    #         mean, var = model(x)
    #         en_mean += mean
    #         en_var += var + mean ** 2

    #     en_mean /= len(ensemble)
    #     en_var /= len(ensemble)
    #     en_var -= en_mean ** 2
    #     return en_mean, en_var



    def ensemble_train(self, epoch, labeled_data, unlabeled_data, optimizer):
        for model in self.clf:
            model.train()
        accFinal = 0.
        data_len = len(labeled_data[0].dataset.X)
        labeled_data = [self.read_data(loader_tr) for loader_tr in labeled_data]
        unlabeled_data = [self.read_data(loader_un,False) for loader_un in unlabeled_data]

        batch_len = int(data_len/self.args.loader_tr_args['batch_size'])
        acc_div = 0.
        # print(batch_len,len(self.clf))
        for idx in range(batch_len):
            for model_idx, model in enumerate(self.clf):
                if model_idx == len(self.clf) - 1:
                    x, y = next(labeled_data[model_idx])
                    acc_div += x.size(0)
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    optimizer[model_idx].zero_grad()
                    y_pred, e1= model(x)
                    accFinal += torch.sum((torch.max(y_pred, 1)[1] == y).float()).data.item()
                    loss = F.cross_entropy(y_pred, y)
                    loss.backward()
                    for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
                    optimizer[model_idx].step()
                else:
                    x, y = next(labeled_data[model_idx])
                    acc_div += x.size(0)
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    # print('y_shape',y.shape)
                    unlabeled_x = next(unlabeled_data[model_idx])
                    unlabeled_x = Variable(unlabeled_x.cuda())
                    optimizer[model_idx].zero_grad()

                    nll = nn.NLLLoss(reduction='mean')
                    y_pred, e1= model(x)
                    accFinal += torch.sum((torch.max(y_pred, 1)[1] == y).float()).data.item()
                    m = nn.LogSoftmax(dim=1)
                    # loss = F.cross_entropy(y_pred, y)
                    nll_loss = nll(m(y_pred), y)
                    # print(unlabeled_x.shape)
                    unlabeled_y, e1 = model(unlabeled_x)
                    # # print('un y_pred', unlabeled_y.shape)
                    v_loss = vat_loss(model, unlabeled_x, unlabeled_y, eps=2.5)
                    loss = v_loss + nll_loss
                    # loss = nll_loss
                    # if 'vatent':
                    #     loss += entropy_loss(unlabeled_y)

                    loss.backward()
                    
                    # clamp gradients, just in case
                    for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
                    optimizer[model_idx].step()
        return accFinal / acc_div

    def train(self, alpha=0.1, n_epoch=10):

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        n_epoch = self.args.n_epoch
        self.clf = [net.apply(weight_reset).cuda() for net in self.net]
        optimizer = [optim.SGD(model.parameters(), lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum) for model in self.clf]

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]

        labeled_data = [DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=self.args.transform_tr), 
                                    shuffle=True,
                                    # pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args) for i in range(len(self.clf))]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_data = [DataLoader(
            self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                         transform=self.args.transform_tr), shuffle=True,
                                    # pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
            **self.args.loader_tr_args) for i in range(len(self.clf))]
        accCurrent = 0.
        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
                # train one epoch
                accCurrent = self.ensemble_train(epoch, labeled_data, unlabeled_data, optimizer)
                test_acc = self.predict(self.X_te, self.Y_te)

                # measure elapsed time
                epoch_time.update(time.time() - ts)

                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Test Accuracy={:.2f}, Train={:.2f}]'.format(recorder.max_accuracy(False),
                                                               recorder.max_accuracy(True)))
                
                
                recorder.update(epoch, -1, accCurrent, 0, test_acc)

                

        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc         

    # def predict(self, X, Y):
    #     # add support for pretrained model
    #     transform=self.args.transform_te if not self.pretrained else self.preprocessing
    #     if type(X) is np.ndarray:
    #         loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
    #                         shuffle=False, **self.args.loader_te_args)
    #     else: 
    #         loader_te = DataLoader(self.handler(X.numpy(), Y, transform=transform), pin_memory=True,
    #                         shuffle=False, **self.args.loader_te_args)
        
    #     for clf in self.clf:
    #         clf.eval()

    #     correct = 0
    #     with torch.no_grad():
    #         for x, y, idxs in loader_te:
    #             x, y = x.to(self.device), y.to(self.device) 
    #             out = self.ensemble_predict(x)
    #             pred = out.max(1)[1]                
    #             correct +=  (y == pred).sum().item() 

    #         test_acc = 1. * correct / len(Y)
   
    #     return test_acc

    def predict(self, X, Y):
        # add support for pretrained model
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                            shuffle=False, **self.args.loader_te_args)
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)
        
        if not self.pretrained:
            self.clf[-1].eval()
        else:
            self.clf[-1].classifier.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf[-1](x)
                pred = out.max(1)[1]                
                correct +=  (y == pred).sum().item() 

            test_acc = 1. * correct / len(Y)

        return test_acc


    def ensemble_predict_prob(self,X, Y):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args.loader_te_args)

        for model in self.clf:
            model.eval()

        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                for idx, model in enumerate(self.clf):
                    if idx == len(self.clf) - 1:
                        continue
                    out, e1 = model(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        return probs

        # f = True
        # for idx, model in enumerate(self.clf):
        #     if idx == len(self.clf) - 1:
        #         continue
        #     y_pred, e1 = model(x)
        #     y_pred = F.softmax(y_pred, dim=1)
        #     if f:
        #         sum_y = y_pred
        #         f = False
        #     else:
        #         sum_y += y_pred
        # return y_pred
        

    def ensemble_predict_var(self,x):
        pred_list = []
        var_list = []
        y_pred = self.ensemble_predict(x)
        pred = y_pred.max(1)[1].data.cpu()
        for idx, model in enumerate(self.clf):
            if idx == len(self.clf) - 1:
                continue
            y_pred, e1 = model(x)
            pred_list.append(y_pred)
        for i in range(y_pred.shape[0]):
            prob = [pred_list[j][i][pred[i]].data.cpu() for j in range(len(self.clf) - 1 )]
            # print(prob)
            var_list.append(np.var(prob))
        # print(x.shape,y_pred.shape,pred_list[0][0][0])
        return var_list

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.ensemble_predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]


        # idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # unlabeled_loader = DataLoader(
        #     self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
        #                  transform=self.args.transform_te), shuffle=True,
        #     **self.args.loader_tr_args)

        # for model in self.clf:
        #     model.eval()
        # P = []
        # probs = self.ensemble_predict_var(x)
        # log_probs = torch.log(probs)
		# U = (probs*log_probs).sum(1)
		# return idxs_unlabeled[U.sort()[1][:n]]
        # with torch.no_grad():
        #     for x, y, idxs in unlabeled_loader:
        #         x, y = Variable(x.cuda()), Variable(y.cuda())
        #         var = self.ensemble_predict_var(x)
        #         # print(idxs.shape)
        #         for i,j in zip(var,idxs):
        #             P.append([i,j])
        # P.sort(key = lambda x : x[0], reverse = True)
        # raw_idx = []
        # for idx, tup in enumerate(P):
        #     if idx < n:
        #         raw_idx.append(tup[1])
        # return idxs_unlabeled[raw_idx]

def adjust_learning_rate(optimizer, epoch, gammas, schedule, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    "Add by YU"
    lr = args.lr
    mu = args.momentum
    for opt in optimizer:
        if args.optimizer != "YF":
            assert len(gammas) == len(
                schedule), "length of gammas and schedule should be equal"
            for (gamma, step) in zip(gammas, schedule):
                if (epoch >= step):
                    lr = lr * gamma
                else:
                    break
            
            for param_group in opt.param_groups:
                param_group['lr'] = lr

        elif args.optimizer == "YF":
            lr = opt._lr
            mu = opt._mu

    return lr, mu
