import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import time
from .strategy import Strategy
from utils import print_log, time_string, AverageMeter, RecorderMeter, convert_secs2time,adjust_learning_rate

class TransformAug:
    # TO DO: implementation of the data augmentation function
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        out1 = self.transform(inp)
        # TO DO: to fit the augmentation transformation
        out2 = 
        return out1, out2


class ssl_UDA(Strategy):
    """
    Our omplementation of the paper: Unsupervised Data Augmentation for Consistency Training
    https://arxiv.org/pdf/1904.12848.pdf
    Google Research, Brain Team, 2 Carnegie Mellon University
    """
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(ssl_UDA, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        """
        n: number of data to query
        return the index of the selected data
        """
        # TO DO: Query the unlabeled data
        unlabeled_data = np.arange(self.n_pool)[~self.idxs_lb]


        # Notice: the returned index should be referenced to the whole training set
        return sel_indexes


    def _train(self, epoch, loader_labeled, loader_unlabeled, optimizer, train_iteration):
        self.clf.train()
        correct = 0.
        total = 0.
        train_loss = 0.
        labeled_train_iter = iter(loader_labeled)
        unlabeled_train_iter = iter(loader_unlabeled)
        # logging 
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()

        for batch_idx in range(int(train_iteration)):
            # TO DO, calculate the loss of the labeled data
            inputs_x, targets_x, _ = labeled_train_iter.next()
            out, e1 = self.clf(inputs_x.cuda())
            
            correct += torch.sum((torch.max(out,1)[1] == targets_x.cuda()).float()).data.item()
            total  += inputs_x.size(0)

            Loss_sup = 

            # TO DO, calculate the loss of the unlabeled data
            (inputs_u,inputs_u2), _ , _ = unlabeled_train_iter.next()
            inputs_u = inputs_u.to(self.device) 
            inputs_u2 = inputs_u2.to(self.device)
            outputs_u,_ = self.clf(inputs_u)
            outputs_u2,_ = self.clf(inputs_u2)

            Loss_unsup = 

            # total loss
            loss = Loss_sup + Loss_unsup 

            train_loss += loss.item()
            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Loss_sup.item(), inputs_x.size(0))
            losses_u.update(Loss_unsup.item(), inputs_x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            if batch_idx % 10 == 0:
                print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return 1. * correct / total, train_loss


    def train(self, alpha=0.1, n_epoch=10):
        # reset the model
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                m.reset_parameters()
        self.clf =  self.net.apply(weight_reset)
        self.clf = nn.DataParallel(self.clf).to(self.device)

        # prepare the training parameters

        parameters = self.clf.parameters() if not self.pretrained \
                                else self.clf.module.classifier.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, 
                                momentum=self.args.momentum)
        
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        previous_loss = 0.
        if idxs_train.shape[0] != 0:

            transform = self.args.transform_tr if not self.pretrained else None

            loader_labeled = DataLoader(self.handler(self.X[idxs_train] if not self.pretrained else self.X_p[idxs_train], 
                                    torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform), shuffle=True, 
                                    **self.args.loader_tr_args)
            loader_unlabeled = DataLoader(self.handler(self.X[idxs_unlabeled] if not self.pretrained else self.X_p[idxs_unlabeled], 
                                    torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(), 
                                    transform=TransformAug(transform)), shuffle=True, 
                                    **self.args.loader_tr_args)
        
            train_iteration = max(len(loader_labeled.dataset.X),
                                len(loader_unlabeled.dataset.X))/self.args.loader_tr_args['batch_size']
            
            # print('has:', n_epoch)
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train(epoch, loader_labeled, loader_unlabeled, optimizer, train_iteration)

                # measure elapsed time
                epoch_time.update(time.time() - ts)

                print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Train Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(True),
                                                               1. - recorder.max_accuracy(True)), self.args.log)
                
                
                recorder.update(epoch, train_los, train_acc, 0, 0)

                # The converge condition 
                if abs(previous_loss - train_los) < 0.001:
                    break
                else:
                    previous_loss = train_los

            self.clf = self.clf.module
        best_train_acc = recorder.max_accuracy(istrain=True)
        return best_train_acc          
        
                       