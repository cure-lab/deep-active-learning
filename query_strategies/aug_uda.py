import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import numpy as np
import time
from .strategy import Strategy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
from torch.nn.functional import kl_div, softmax, log_softmax
import random
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
from copy import deepcopy

tsa = False
uda_confidence_thresh = -1
uda_softmax_temp = 0.4


def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output


class RandAugment(object):
    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),

            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),

            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),

            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.8, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),

            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor)
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }
        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            # "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: PIL.ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: PIL.ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: PIL.ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: PIL.ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: PIL.ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: PIL.ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: PIL.ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: PIL.ImageOps.equalize(img),
            "invert": lambda img, magnitude: PIL.ImageOps.invert(img)
        }
        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


class TransformUDA(object):
    def __init__(self, mean, std, size, channel):
        self.org = transforms.Compose([
            transforms.RandomCrop(size=size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        self.aug = transforms.Compose([
            transforms.RandomCrop(size=size, padding=4),
            transforms.RandomHorizontalFlip(),
            RandAugment((128,) if channel==1 else (128, 128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def __call__(self, x):
        original_img = self.org(x)
        augment_img = self.aug(x)
        return original_img, augment_img


class uda(Strategy):
    """
    Our omplementation of the paper: Unsupervised Data Augmentation for Consistency Training
    https://arxiv.org/pdf/1904.12848.pdf
    Google Research, Brain Team, 2 Carnegie Mellon University
    """

    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(uda, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def query(self, n):
        """
        n: number of data to query
        return the index of the selected data
        """
        # TO DO: Query the unlabeled data
        inds = np.where(self.idxs_lb == 0)[0]
        # Notice: the returned index should be referenced to the whole training set
        return inds[np.random.permutation(len(inds))][:n]

    def _train(self, epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer, global_batch):
        self.clf.train()
        accFinal = 0.
        train_loss = 0.
        total_steps = 36000
        iter_unlabeled = iter(loader_tr_unlabeled)
        for batch_idx, (x, y, idxs) in enumerate(loader_tr_labeled):
            y = y.to(self.device)
            try:
                (inputs_u, inputs_u2), _, _ = next(iter_unlabeled)
            except StopIteration:
                iter_unlabeled = iter(loader_tr_unlabeled)
                (inputs_u, inputs_u2), _, _ = next(iter_unlabeled)
            input_all = torch.cat([x, inputs_u, inputs_u2]).to(self.device)
            output_all, _ = self.clf(input_all)
            output_sup = output_all[:len(x)]
            cross_entropy = nn.CrossEntropyLoss(reduction="none")
            loss = cross_entropy(output_sup, y)     # loss for supervised learning
            if tsa:
                global_step = global_batch * 20 + batch_idx + 1
                tsa_thresh = get_tsa_thresh(tsa, global_step, total_steps, start=0.1, end=1).to(self.device)
                one_hot_labels = torch.zeros(256, 10).scatter_(1, y, 1).type(torch.float32)
                larger_than_threshold = torch.exp(-loss) > tsa_thresh
                loss_mask = torch.ones_like(y, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
                loss = torch.sum(loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1),
                                                                       torch.tensor(1.).to(self.device))
            else:
                loss = torch.mean(loss)

            output_unsup = output_all[len(x):]
            output_u, output_u2 = torch.chunk(output_unsup, 2)
            output_u = softmax(output_u, dim=1).detach()
            output_u2 = log_softmax(output_u2, dim=1)
            kl = torch.sum(kl_div(output_u2, output_u, reduction='none'), dim=1)    # loss for unsupervised learning
            if uda_confidence_thresh != -1:
                unsup_loss_mask = torch.max(output_u, dim=1)[0] > uda_confidence_thresh
                unsup_loss_mask = unsup_loss_mask.type(torch.float32).to(self.device)
                unsup_loss = torch.sum(kl * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                 torch.tensor(1.).to(self.device))
                loss += unsup_loss
            else:
                loss += torch.mean(kl)

            train_loss += loss.item()
            accFinal += torch.sum((torch.max(output_sup, 1)[1] == y).float()).data.item()


            nan_mask = torch.isnan(x)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())

            nan_mask_out = torch.isnan(y)
            if nan_mask_out.any():
                raise RuntimeError(f"Found NAN in output indices: ", nan_mask.nonzero())

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
        # if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
        # find_unused_parameters=True,
        # )
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr=self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0
        train_acc = 0.
        global_batch = 0
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
                                           **self.args.loader_tr_args)
        if idxs_unlabeled.shape[0] != 0:
            transform = self.args.transform_te
            mean = self.args.normalize['mean']
            std = self.args.normalize['std']
            train_data_unlabeled = self.handler(self.X[idxs_unlabeled],
                                                torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                                                transform=TransformUDA(mean=mean, std=std, size=self.args.img_size,channel=self.args.channels))
            loader_tr_unlabeled = DataLoader(train_data_unlabeled,
                                             shuffle=True,
                                             pin_memory=True,
                                             # sampler = DistributedSampler(train_data),
                                             worker_init_fn=self.seed_worker,
                                             generator=self.g,
                                             **self.args.loader_tr_args)
            
            print(self.args.loader_tr_args)
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule,
                                                                self.args)

                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins,
                                                                     need_secs)

                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr_labeled, loader_tr_unlabeled, optimizer, global_batch)
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
            global_batch += 1
            if self.args.save_model:
                self.save_model()
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module
            # self.save_tta_values(self.get_tta_values())

        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc