import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import os
import numpy as np
import time
from .strategy import Strategy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
from .aug_uda import TransformUDA
from copy import deepcopy

unsup_ratio = 7
p_cutoff = 0.95


class TransformWeak(object):
    def __init__(self, mean, std, size):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=size,
                                  padding=int(size * 0.125),
                                  padding_mode='reflect')])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        return self.normalize(weak)


class pseudolabel(Strategy):
    """
    Our omplementation of the paper: Unsupervised Data Augmentation for Consistency Training
    https://arxiv.org/pdf/1904.12848.pdf
    Google Research, Brain Team, 2 Carnegie Mellon University
    """

    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(pseudolabel, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def query(self, n):
        """
        n: number of data to query
        return the index of the selected data
        """
        # TO DO: Query the unlabeled data
        inds = np.where(self.idxs_lb == 0)[0]
        # Notice: the returned index should be referenced to the whole training set
        return inds[np.random.permutation(len(inds))][:n]

    def _train(self, epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer):
        self.clf.train()
        accFinal = 0.
        train_loss = 0.
        iter_unlabeled = iter(loader_tr_unlabeled)
        for batch_idx, (x, y, idxs) in enumerate(loader_tr_labeled):
            y = y.to(self.device)
            try:
                (inputs_u), _, idx = next(iter_unlabeled)
            except StopIteration:
                iter_unlabeled = iter(loader_tr_unlabeled)
                (inputs_u), _, idx = next(iter_unlabeled)


            logits_x_lb, _ = self.clf(x)
            logits_x_ulb_w, _ = self.clf(inputs_u)
            loss = F.nll_loss(F.log_softmax(logits_x_lb, dim=-1), y, reduction='mean')   # loss for supervised learning

            pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff).float()
            unsup_loss = (F.nll_loss(F.log_softmax(logits_x_ulb_w, dim=-1), max_idx.detach(), reduction='none') * mask).mean()

            loss += unsup_loss
            train_loss += loss.item()
            accFinal += torch.sum((torch.max(logits_x_lb, 1)[1] == y).float()).data.item()

            # exit()
            optimizer.zero_grad()
            loss.backward()

            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()

            if batch_idx % 10 == 0:
                print("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / len(loader_tr_labeled.dataset.X), train_loss

    def train(self, alpha=0.1, n_epoch=10):
        self.clf =  deepcopy(self.net)
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr=self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        previous_loss = 0.

        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr

            train_data_labeled = self.handler(self.X[idxs_train],
                                              torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                              transform=transform)
            loader_tr_labeled = DataLoader(train_data_labeled,
                                           shuffle=True,
                                           pin_memory=True,
                                           # sampler = DistributedSampler(train_data),
                                           worker_init_fn=self.seed_worker,
                                           generator=self.g,
                                           **{'batch_size': 250, 'num_workers': 1})
        if idxs_unlabeled.shape[0] != 0:
            mean = self.args.normalize['mean']
            std = self.args.normalize['std']
            train_data_unlabeled = self.handler(self.X[idxs_unlabeled],
                                                torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                                                transform=TransformWeak(mean=mean, std=std, size=self.args.img_size))
            loader_tr_unlabeled = DataLoader(train_data_unlabeled,
                                             shuffle=True,
                                             pin_memory=True,
                                             # sampler = DistributedSampler(train_data),
                                             worker_init_fn=self.seed_worker,
                                             generator=self.g,
                                             **{'batch_size': int(250 * unsup_ratio), 'num_workers': 1})
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule,
                                                                self.args)

                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins,
                                                                     need_secs)

                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer)
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
                if abs(previous_loss - train_los) < 0.0005:
                    break
                else:
                    previous_loss = train_los
            if self.args.save_model:
                self.save_model()
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module
            # self.save_tta_values(self.get_tta_values())


        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc
