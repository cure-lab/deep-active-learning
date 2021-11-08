import torch
import torch.nn.functional as F
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import numpy as np
import time
from .strategy import Strategy
from utils import print_log, time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate

import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2470, 0.2435, 0.2616)


class UDALoss(nn.Module):
    def __init__(self):
        super(UDALoss, self).__init__()

    def forward(self, model, x, x_h):
        batchsize = x.shape[0]
        with torch.no_grad():
            output, aux = model(x)
            pred_x = F.softmax(output, dim=1)
        output, aux = model(x_h)
        pred_x_h = F.log_softmax(output, dim=1)
        lds = F.kl_div(pred_x_h, pred_x, None, None, reduction='sum') / batchsize
        return lds


class RandAugment(object):
    # TO DO: implementation of the data augmentation function

    def AutoContrast(self, img, **kwarg):
        return PIL.ImageOps.autocontrast(img)

    def Brightness(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Brightness(img).enhance(v)

    def Color(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Color(img).enhance(v)

    def Contrast(self, img, v, max_v, bias=0):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Contrast(img).enhance(v)

    def CutoutConst(self, img, v, max_v, **kwarg):
        v = int(v * max_v / 10)
        w, h = img.size
        x0 = np.random.uniform(0, w)
        y0 = np.random.uniform(0, h)
        x0 = int(max(0, x0 - v / 2.))
        y0 = int(max(0, y0 - v / 2.))
        x1 = int(min(w, x0 + v))
        y1 = int(min(h, y0 + v))
        xy = (x0, y0, x1, y1)
        # gray
        color = (127, 127, 127)
        img = img.copy()
        PIL.ImageDraw.Draw(img).rectangle(xy, color)
        return img

    def Equalize(self, img, **kwarg):
        return PIL.ImageOps.equalize(img)

    def Identity(self, img, **kwarg):
        return img

    def Invert(self, img, **kwarg):
        return PIL.ImageOps.invert(img)

    def Posterize(self, img, v, max_v, bias, **kwarg):
        v = int(v * max_v / 10) + bias
        return PIL.ImageOps.posterize(img, v)

    def Rotate(self, img, v, max_v, **kwarg):
        v = float(v) * max_v / 10
        if random.random() < 0.5:
            v = -v
        return img.rotate(v)

    def Sharpness(self, img, v, max_v, bias):
        v = float(v) * max_v / 10 + bias
        return PIL.ImageEnhance.Sharpness(img).enhance(v)

    def ShearX(self, img, v, max_v, **kwarg):
        v = float(v) * max_v / 10
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), RESAMPLE_MODE)

    def ShearY(self, img, v, max_v, **kwarg):
        v = float(v) * max_v / 10
        if random.random() < 0.5:
            v = -v
        return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), RESAMPLE_MODE)

    def Solarize(self, img, v, max_v, **kwarg):
        v = int(v * max_v / 10)
        return PIL.ImageOps.solarize(img, 256 - v)

    def __init__(self, n, m, resample_mode=PIL.Image.BILINEAR):
        assert n >= 1
        assert m >= 1
        global RESAMPLE_MODE
        RESAMPLE_MODE = resample_mode
        self.n = n
        self.m = m
        self.augment_pool = [(self.AutoContrast, None, None),
                             (self.Brightness, 1.8, 0.1),
                             (self.Color, 1.8, 0.1),
                             (self.Contrast, 1.8, 0.1),
                             (self.CutoutConst, 40, None),
                             (self.Equalize, None, None),
                             (self.Invert, None, None),
                             (self.Posterize, 4, 0),
                             (self.Rotate, 30, None),
                             (self.Sharpness, 1.8, 0.1),
                             (self.ShearX, 0.3, None),
                             (self.ShearY, 0.3, None),
                             (self.Solarize, 256, None)
                             ]

    def __call__(self, inp):
        # out1 = self.transform(inp)
        # TO DO: to fit the augmentation transformation
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                inp = op(inp, v=self.m, max_v=max_v, bias=bias)
        return inp


class TransformRandAug(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            RandAugment(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, inp):
        weak = self.weak(inp)
        strong = self.strong(inp)
        return self.normalize(weak), self.normalize(strong)


class ssl_RandUDA(Strategy):
    """
    Our omplementation of the paper: Unsupervised Data Augmentation for Consistency Training
    https://arxiv.org/pdf/1904.12848.pdf
    Google Research, Brain Team, 2 Carnegie Mellon University
    """

    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(ssl_RandUDA, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        """
        n: number of data to query
        return the index of the selected data
        """
        # TO DO: Query the unlabeled data
        inds = np.where(self.idxs_lb == 0)[0]
        # Notice: the returned index should be referenced to the whole training set
        return inds[np.random.permutation(len(inds))][:n]

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
            try:
                inputs_x, targets_x, _ = labeled_train_iter.next()
            except StopIteration:
                batch_iterator = iter(loader_labeled)
                inputs_x, targets_x, _ = batch_iterator.next()
            inputs_x = inputs_x.to(self.device)
            targets_x = targets_x.to(self.device)
            output, e1 = self.clf(inputs_x)
            cross_entropy = nn.CrossEntropyLoss()
            Loss_sup = cross_entropy(output, targets_x)

            correct += torch.sum((torch.max(output, 1)[1] == targets_x).float()).data.item()
            total += inputs_x.size(0)

            # TO DO, calculate the loss of the unlabeled data
            try:
                (inputs_u, inputs_u2), _, _ = unlabeled_train_iter.next()
            except StopIteration:
                batch_iterator = iter(unlabeled_train_iter)
                (inputs_u, inputs_u2), _, _ = batch_iterator.next()
            inputs_u = inputs_u.to(self.device)
            inputs_u2 = inputs_u2.to(self.device)
            outputs_u, _ = self.clf(inputs_u)
            outputs_u2, _ = self.clf(inputs_u2)

            uda_loss = UDALoss()
            Loss_unsup = uda_loss(self.clf, inputs_u, inputs_u2)

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
                print("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return 1. * correct / total, train_loss

    def train(self, alpha=0.1, n_epoch=10):
        # reset the model
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        self.clf = self.clf.apply(weight_reset)
        self.clf = nn.DataParallel(self.clf).to(self.device)

        # prepare the training parameters

        parameters = self.clf.parameters() if not self.pretrained \
            else self.clf.module.classifier.parameters()
        optimizer = optim.SGD(parameters, lr=self.args.lr, weight_decay=5e-4,
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

            loader_labeled = DataLoader(
                self.handler(self.X[idxs_train] if not self.pretrained else self.X_p[idxs_train],
                             torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                             transform=transform), shuffle=True,
                **self.args.loader_tr_args)
            loader_unlabeled = DataLoader(
                self.handler(self.X[idxs_unlabeled] if not self.pretrained else self.X_p[idxs_unlabeled],
                             torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                             transform=TransformRandAug(cifar10_mean, cifar10_std)), shuffle=True,
                **self.args.loader_tr_args)

            train_iteration = max(len(loader_labeled.dataset.X),
                                  len(loader_unlabeled.dataset.X)) / self.args.loader_tr_args['batch_size']

            # print('has:', n_epoch)
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule,
                                                                self.args)

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
                                                                                   1. - recorder.max_accuracy(True)),
                          self.args.log)

                recorder.update(epoch, train_los, train_acc, 0, 0)

                # The converge condition
                if abs(previous_loss - train_los) < 0.001:
                    break
                else:
                    previous_loss = train_los

            self.clf = self.clf.module
        best_train_acc = recorder.max_accuracy(istrain=True)
        return best_train_acc

