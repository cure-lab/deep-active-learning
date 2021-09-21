import numpy as np
import random
import sys
import gzip

from torchvision.transforms.transforms import Resize
import openml
import os
import argparse
from dataset import get_dataset, get_handler, get_wa_handler
from sklearn.preprocessing import LabelEncoder
import torch.nn.functional as F
from torch import nn
from torchvision import transforms
import torch
import pdb
from scipy.stats import zscore
from coverage import *
import csv
import time

import query_strategies 
import mymodels

def print_log(string, log):
    print (string)
    with open(log, 'w+') as f:
        f.write(string)

query_strategies_name = sorted(name for name in query_strategies.__dict__
                     if callable(query_strategies.__dict__[name]))

# code based on https://github.com/ej0cl6/deep-active-learning"
parser = argparse.ArgumentParser()
parser.add_argument('--strategy', help='acquisition algorithm', type=str, choices=query_strategies_name, 
                    default='rand')
parser.add_argument('--did', help='openML dataset index, if any', type=int, default=0)
parser.add_argument('--lr', help='learning rate', type=float, default=1e-4)
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str, default='mlp')
parser.add_argument('--data_path', help='data path', type=str, default='/research/dept2/yuli/datasets')
parser.add_argument('--dataset', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--nQuery', help='number of points to query in a batch', type=int, default=100)
parser.add_argument('--nStart', help='number of points to start', type=int, default=100)
parser.add_argument('--nEnd', help = 'total number of points to query', type=int, default=50000)
parser.add_argument('--nEmb', help='number of embedding dims (mlp)', type=int, default=256)
parser.add_argument('--rand_idx', help='the index of the repeated experiments', type=int, default=1)
parser.add_argument('--save_path', help='result save save_dir', default='./save')
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
args = parser.parse_args()


if not os.path.isdir(args.save_path):
    os.makedirs(args.save_path)
log = os.path.join(args.save_path,
                    'log_seed_{}.txt'.format(args.manualSeed))

# non-openml data defaults
args_pool = {'mnist':
                {'n_epoch': 20,  
                 'num_class':10,
                'channels':1,
                 'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'fashionmnist':
                {'n_epoch': 20, 
                 'num_class':10,
                'channels':1,
                'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'svhn':
                {'n_epoch': 200, 
                 'num_class':10,
                'channels':3,
                'transform_tr': transforms.Compose([ 
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
            'cifar10':
                {'n_epoch': 150, 
                 'num_class':10,
                 'channels':3,
                 'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.3}},
            'gtsrb': 
               {'n_epoch': 80, 
                 'num_class':43,
                'channels':3,
                 'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                 'transform_te': transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 1},
                 'optimizer_args':{'lr': 0.01, 'momentum': 0.3}}
        }


if not os.path.exists(args.data_path):
    os.makedirs(args.data_path)


# load openml dataset if did is supplied
if args.did > 0:
    openml.config.apikey = '3411e20aff621cc890bf403f104ac4bc'
    openml.config.set_cache_directory(args.path)
    ds = openml.datasets.get_dataset(args.did)
    data = ds.get_data(target=ds.default_target_attribute)
    X = np.asarray(data[0])
    y = np.asarray(data[1])
    y = LabelEncoder().fit(y).transform(y)

    args.nClasses = int(max(y) + 1)
    nSamps, args.dim = np.shape(X)
    testSplit = .1
    inds = np.random.permutation(nSamps)
    X = X[inds]
    y = y[inds]

    split =int((1. - testSplit) * nSamps)
    while True:
        inds = np.random.permutation(split)
        if len(inds) > 50000: inds = inds[:50000]
        X_tr = X[:split]
        X_tr = X_tr[inds]
        X_tr = torch.Tensor(X_tr)

        y_tr = y[:split]
        y_tr = y_tr[inds]
        Y_tr = torch.Tensor(y_tr).long()

        X_te = torch.Tensor(X[split:])
        Y_te = torch.Tensor(y[split:]).long()

        if len(np.unique(Y_tr)) == args.nClasses: break


    args = {'transform_tr':transforms.Compose([transforms.ToTensor()]),
            'n_epoch':10,
            'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
            'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
            'optimizer_args':{'lr': 0.01, 'momentum': 0},
            'transform_te':transforms.Compose([transforms.ToTensor()])}
    handler = get_handler('other')

# load non-openml dataset
else:
    X_tr, Y_tr, X_te, Y_te = get_dataset(args.dataset, args.data_path)
    args.dim = np.shape(X_tr)[1:]
    handler = get_handler(args.dataset)


# start experiment
n_pool = len(Y_tr)
n_test = len(Y_te)

# parameters
args.nEnd =  args.nEnd if args.nEnd != -1 else n_pool
args.nQuery = args.nQuery if args.nQuery != -1 else (n_pool - args.nStart)
NUM_INIT_LB = args.nStart
NUM_QUERY = args.nQuery
NUM_ROUND = int((args.nEnd - NUM_INIT_LB)/ args.nQuery)

print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
print('number of testing pool: {}'.format(n_test), flush=True)

# generate initial labeled pool
idxs_lb = np.zeros(n_pool, dtype=bool)
idxs_tmp = np.arange(n_pool)
np.random.shuffle(idxs_tmp)
idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True




dataset_args = args_pool[args.dataset]
dataset_args['pretrained'] = False

# load specified network
if args.model == 'lenet':
    net = mymodels.LeNet(n_class=dataset_args['num_class'])
elif args.model == 'resnet':
    net = mymodels.ResNet18(num_classes=dataset_args['num_class'])
else: 
    print('choose a valid model - mlp, resnet, or ...(not implemented now)', flush=True)
    raise ValueError

if args.did > 0 and args.model != 'mlp':
    print('openML datasets only work with mlp', flush=True)
    raise ValueError

if type(X_tr[0]) is not np.ndarray:
    X_tr = X_tr.numpy()


if args.strategy == 'ActiveLearningByLearning': # active learning by learning (albl)
    albl_list = [query_strategies.LeastConfidence(X_tr, Y_tr, idxs_lb, net, handler, dataset_args),
                    query_strategies.CoreSet(X_tr, Y_tr, idxs_lb, net, handler, dataset_args)]
    strategy = query_strategies.ActiveLearningByLearning(X_tr, Y_tr, idxs_lb, net, handler, dataset_args, 
                strategy_list=albl_list, delta=0.1)
elif args.strategy == 'WAAL': # waal
    test_handler = handler
    train_handler = get_wa_handler(args.dataset)
    strategy =  query_strategies.WAAL(X_tr, Y_tr, idxs_lb, net, 
                                        train_handler, test_handler, dataset_args)
else:
    strategy = query_strategies.__dict__[args.strategy](X_tr, Y_tr, idxs_lb, net, handler, 
                        args=dataset_args)

print_log('Strategy successfully loaded...', log)
# print info
if args.did > 0: DATA_NAME='OML' + str(args.did)

print_log("dataset: {}".format(args.dataset), log)
print_log("strategy: {}".format(strategy), log)

# round 0 accuracy
alpha = 2e-3
strategy.train(alpha=alpha, n_epoch=dataset_args['n_epoch'])
P = strategy.predict(X_te, Y_te)
acc = np.zeros(NUM_ROUND+1)
acc[0] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
print(str(args.nStart) + '\ttesting accuracy {}'.format(acc[0]), flush=True)

out_file = os.path.join(args.save_path, 'main_result.csv')
for rd in range(1, NUM_ROUND+1):
    
    print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)

    # query
    ts = time.time()
    output = strategy.query(NUM_QUERY)
    q_idxs = output
    idxs_lb[q_idxs] = True
    te = time.time()
    tp = te - ts
    # report weighted accuracy
    corr = (strategy.predict(X_tr[q_idxs], torch.Tensor(Y_tr.numpy()[q_idxs]).long())).numpy() == Y_tr.numpy()[q_idxs]

    # update
    strategy.update(idxs_lb)
    strategy.train(alpha=alpha, n_epoch=dataset_args['n_epoch'])

    # round accuracy
    P = strategy.predict(X_te, Y_te)
    acc[rd] = 1.0 * (Y_te == P).sum().item() / len(Y_te)
    print_log(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), log)

    print_log("logging...", log)
    with open(out_file, 'a+') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([
                        args.strategy,
                        args.rand_idx,
                        'budget',
                        args.nEnd,
                        'nQuery',
                        args.nQuery,
                        'currentQuery',
                        args.nQuery*rd,
                        'accCompare',
                        acc[0],
                        acc[rd],
                        acc[rd] - acc[0],
                        'coresetCov',
                        coverage, 
                        'timePerIter',
                        tp
                        ])
    
    if sum(~strategy.idxs_lb) < args.nQuery: 
        print('too few remaining points to query')
        break

print_log('success!', log)
