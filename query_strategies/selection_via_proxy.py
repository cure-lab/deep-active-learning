import torch
import numpy as np
from .strategy import Strategy
import mymodels
from sklearn import preprocessing
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import time

# Implementation of the paper: Selection via Proxy: Efficient Data Selection for Deep Learning
# Published in ICLR'2020
# Code: https://github.com/stanford-futuredata/selection-via-proxy/blob/master/svp/cifar/active.py



class Proxy(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(Proxy, self).__init__(X, Y,  X_te, Y_te, idxs_lb, net, handler, args)
        self.proxy_model = mymodels.__dict__[args.proxy_model](n_class=args.n_class)

    def _train(self, epoch, loader_tr, optimizer):
        model = self.proxy_model if self.args.proxy_model is not None else self.clf
        
        model.train()

        accFinal = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device) 
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())

            # exit()
            optimizer.zero_grad()

            out, e1 = model(x)
            nan_mask_out = torch.isnan(y)
            if nan_mask_out.any():
                raise RuntimeError(f"Found NAN in output indices: ", nan_mask.nonzero())
                
            loss = F.cross_entropy(out, y)

            train_loss += loss.item()
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            
            loss.backward()
            
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, model.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()

            if batch_idx % 10 == 0:
                print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / len(loader_tr.dataset.X), train_loss


    def train(self, alpha=0.1, n_epoch=10):
        n_query = int(self.args.nQuery*len(self.Y)/100)
        if self.idxs_lb.sum() + n_query > (self.args.nEnd*len(self.Y)/100):
            # the last round, we use the original model for training
            self.args.proxy_model = None

        model = self.proxy_model if self.args.proxy_model is not None else self.clf

        # train the proxy model for query
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                m.reset_parameters()
        
        model =  model.apply(weight_reset)
        model = nn.DataParallel(model).to(self.device)
        parameters = model.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        previous_loss = 0.
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr

            loader_tr = DataLoader(self.handler(self.X[idxs_train], 
                                    torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform), shuffle=True, 
                                    **self.args.loader_tr_args)

            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train_proxy(epoch, loader_tr, optimizer)
                test_acc = self.predict(self.X_te, self.Y_te)

                # measure elapsed time
                epoch_time.update(time.time() - ts)

                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Test Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               1. - recorder.max_accuracy(False)))
                
                
                recorder.update(epoch, train_los, train_acc, 0, test_acc)

                # The converge condition 
                if abs(previous_loss - train_los) < 0.0001:
                    break
                else:
                    previous_loss = train_los

            model = model.module
        best_train_acc = recorder.max_accuracy(istrain=True)
        return best_train_acc                


    def predict(self, X, Y):
        model = self.proxy_model if self.args.proxy_model is not None else self.clf
        transform=self.args.transform_te
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        
        self.clf.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = model(x)
                pred = out.max(1)[1]                
                correct +=  (y == pred).sum().item() 

            test_acc = 1. * correct / len(Y)
   
        return test_acc

    def predict_prob(self, X, Y):
        model = self.proxy_model 

        transform = self.args.transform_te
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, **self.args.loader_te_args)
        self.clf.eval()

        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs
    
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        log_probs = torch.log(probs)
        U = (probs*log_probs).sum(1)
        return idxs_unlabeled[U.sort()[1][:n]]


