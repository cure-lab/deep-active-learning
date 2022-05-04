from re import I
import numpy as np
import random
import sys

import os
import argparse
from dataset import get_dataset, get_handler, get_wa_handler
import torch.nn.functional as F
from torchvision import transforms
import torch
import csv
import time

import query_strategies 
import mymodels
from utils import print_log
# import torch.distributed as dist

# code based on https://github.com/ej0cl6/deep-active-learning"
os.environ['CUBLAS_WORKSPACE_CONFIG']= ':16:8'

query_strategies_name = sorted(name for name in query_strategies.__dict__
                     if callable(query_strategies.__dict__[name]))
model_name = sorted(name for name in mymodels.__dict__)


###############################################################################
parser = argparse.ArgumentParser()
# strategy
parser.add_argument('--strategy', help='acquisition algorithm', type=str, choices=query_strategies_name, 
                    default='rand')
parser.add_argument('--nQuery',  type=float, default=1,
                    help='number of points to query in a batch (%)')
parser.add_argument('--nStart', type=float, default=10,
                    help='number of points to start (%)')
parser.add_argument('--nEnd',type=float, default=100,
                        help = 'total number of points to query (%)')
parser.add_argument('--nEmb',  type=int, default=256,
                        help='number of embedding dims (mlp)')
parser.add_argument('--rand_idx', type=int, default=1,
                    help='the index of the repeated experiments', )
parser.add_argument('--manualSeed', type=int, default=None, help='manual seed')
parser.add_argument("-t","--full", action='store_true',
                    help="Training on the entire dataset")

# model and data
parser.add_argument('--model', help='model - resnet, vgg, or mlp', type=str)
parser.add_argument('--dataset', help='dataset (non-openML)', type=str, default='')
parser.add_argument('--data_path', help='data path', type=str, default='')
parser.add_argument('--save_path', help='result save save_dir', default='./save')
parser.add_argument('--save_file', help='result save save_dir', default='result.csv')

# for gcn, designed for uncertainGCN and coreGCN
parser.add_argument("-n","--hidden_units", type=int, default=128,
                    help="Number of hidden units of the graph")
parser.add_argument("-r","--dropout_rate", type=float, default=0.3,
                    help="Dropout rate of the graph neural network")
parser.add_argument("-l","--lambda_loss",type=float, default=1.2, 
                    help="Adjustment graph loss parameter between the labeled and unlabeled")
parser.add_argument("-s","--s_margin", type=float, default=0.1,
                    help="Confidence margin of graph")

# for ensemble based methods
parser.add_argument('--n_ensembles', type=int, default=1, 
                    help='number of ensemble')

# for proxy based selection
parser.add_argument('--proxy_model', type=str, default=None,
                    help='the architecture of the proxy model')


# training hyperparameters
parser.add_argument('--optimizer',
                    type=str,
                    default='SGD',
                    choices=['SGD', 'Adam', 'YF'])
parser.add_argument('--n_epoch', type=int, default=100,
                    help='number of training epochs in each iteration')
parser.add_argument('--schedule',
                    type=int,
                    nargs='+',
                    default=[80, 120],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate.')
parser.add_argument('--gammas',
                    type=float,
                    nargs='+',
                    default=[0.1, 0.1],
                    help=
                    'LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
parser.add_argument('--pretrained', 
                    action='store_true',
                    default=False, help='use pretrained feature extractor')
parser.add_argument('--save_model', 
                    action='store_true',
                    default=False, help='save model every steps')
parser.add_argument('--save_tta', 
                    action='store_true',
                    default=False, help='save tta every epochs')
parser.add_argument('--is_load_ckpt', 
                    action='store_true',
                    help='load model from memory, True or False')

# automatically set
# parser.add_argument("--local_rank", type=int)

##########################################################################
args = parser.parse_args()
# set the backend of the distributed parallel
# ngpus = torch.cuda.device_count()
# dist.init_process_group("nccl")

############################# For reproducibility #############################################

if args.manualSeed is None:
    args.manualSeed = args.rand_idx
    # args.manualSeed = random.randint(0, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
np.random.seed(args.manualSeed)

if torch.cuda.is_available():
    torch.cuda.manual_seed(args.manualSeed)
    # True ensures the algorithm selected by CUFA is deterministic
    # torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
    # False ensures CUDA select the same algorithm each time the application is run
    torch.backends.cudnn.benchmark = False

############################# Specify the hyperparameters #######################################
 
args_pool = {'mnist':
                { 
                 'n_class':10,
                 'channels':1,
                 'size': 28,
                 'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                },
            'fashionmnist':
                {
                 'n_class':10,
                'channels':1,
                'size': 28,
                'transform_tr': transforms.Compose([
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(), 
                                transforms.Normalize((0.1307,), (0.3081,))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1307,), (0.3081,))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 1},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 1},
                 'normalize':{'mean': (0.1307,), 'std': (0.3081,)},
                },
            'svhn':
                {
                 'n_class':10,
                'channels':3,
                'size': 32,
                'transform_tr': transforms.Compose([ 
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                 'loader_tr_args':{'batch_size': 128, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': (0.4377, 0.4438, 0.4728), 'std': (0.1980, 0.2010, 0.1970)},
                },
            'cifar10':
                {
                 'n_class':10,
                 'channels':3,
                 'size': 32,
                 'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'transform_te': transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                 'normalize':{'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2470, 0.2435, 0.2616)},
                 },
            'gtsrb': 
               {
                 'n_class':43,
                 'channels':3,
                 'size': 32,
                 'transform_tr': transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.RandomCrop(size = 32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                 'transform_te': transforms.Compose([
                                    transforms.Resize((32, 32)),
                                    transforms.ToTensor(), 
                                    transforms.Normalize([0.3337, 0.3064, 0.3171], [0.2672, 0.2564, 0.2629])]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 1024, 'num_workers': 8},
                 'normalize':{'mean': [0.3337, 0.3064, 0.3171], 'std': [0.2672, 0.2564, 0.2629]},
                },
            'tinyimagenet': 
               {
                'n_class':200,
                'channels':3,
                'size': 64,
                'transform_tr': transforms.Compose([
                                    transforms.RandomCrop(size = 64, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 'transform_te': transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]),
                 'loader_tr_args':{'batch_size': 256, 'num_workers': 8},
                 'loader_te_args':{'batch_size': 512, 'num_workers': 8},
                 'normalize':{'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
                }
        }

###############################################################################
###############################################################################

def main():
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    

    log = os.path.join(args.save_path,
                        'log_seed_{}.txt'.format(args.manualSeed))

    # print the args
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(str(state), log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    # load the dataset specific parameters
    dataset_args = args_pool[args.dataset]
    args.n_class = dataset_args['n_class']
    args.img_size = dataset_args['size']
    args.channels = dataset_args['channels']
    args.transform_tr = dataset_args['transform_tr']
    args.transform_te = dataset_args['transform_te']
    args.loader_tr_args = dataset_args['loader_tr_args']
    args.loader_te_args = dataset_args['loader_te_args']
    args.normalize = dataset_args['normalize']
    args.log = log 

    # load dataset
    X_tr, Y_tr, X_te, Y_te = get_dataset(args.dataset, args.data_path)
    if type(X_tr[0]) is not np.ndarray:
        X_tr = X_tr.numpy()
        X_te = X_te.numpy()
        
    args.dim = np.shape(X_tr)[1:]
    handler = get_handler(args.dataset)

    n_pool = len(Y_tr)
    n_test = len(Y_te)

    # parameters
    if args.dataset == 'mnist':
        args.schedule = [20, 40]

    args.nEnd =  args.nEnd if args.nEnd != -1 else 100
    args.nQuery = args.nQuery if args.nQuery != -1 else (args.nEnd - args.nStart)
    if args.full:
        # train the model with full data
        NUM_INIT_LB = n_pool
        NUM_QUERY = 0
        NUM_ROUND = 0
    else:
        # train the model iteratively
        NUM_INIT_LB = int(args.nStart*n_pool/100)
        NUM_QUERY = int(args.nQuery*n_pool/100)
        NUM_ROUND = int((int(args.nEnd*n_pool/100) - NUM_INIT_LB)/ NUM_QUERY) 
        if (int(args.nEnd*n_pool/100) - NUM_INIT_LB)% NUM_QUERY != 0:
            NUM_ROUND += 1
    
    print_log("[init={:02d}] [query={:02d}] [end={:02d}]".format(NUM_INIT_LB, NUM_QUERY, int(args.nEnd*n_pool/100)), log)

    # load specified network
    if args.strategy == 'ensemble':
        net = [mymodels.__dict__[args.model](n_class=args.n_class) for _ in range(args.n_ensembles)]
    elif 'ssl_' in args.strategy:
        net = [mymodels.__dict__[args.model](n_class=args.n_class) for _ in range(2)]
    else:
        net = mymodels.__dict__[args.model](n_class=args.n_class)

    # if args.load_model:
    #     idxs_lb = np.load(os.path.join(args.save_path, args.strategy+'_'+args.model+'_'+args.load_model+'_'+str(args.manualSeed)+'.npy'))
    #     if len(np.arange(n_pool)[idxs_lb]) != NUM_INIT_LB:
    #         raise ValueError("Number of labeled data should equal to start number")
        
    # else:
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
        

    # selection strategy
    if args.strategy == 'ActiveLearningByLearning': # active learning by learning (albl)
        albl_list = [query_strategies.LeastConfidence(X_tr, Y_tr, X_te, Y_te, idxs_lb, net, handler, args),
                        query_strategies.CoreSet(X_tr, Y_tr, X_te, Y_te, idxs_lb, net, handler, args)]
        strategy = query_strategies.ActiveLearningByLearning(X_tr, Y_tr, X_te, Y_te, idxs_lb, net, handler, args, 
                    strategy_list=albl_list, delta=0.1)
    elif args.strategy == 'WAAL': # waal
        test_handler = handler
        train_handler = get_wa_handler(args.dataset)
        strategy =  query_strategies.WAAL(X_tr, Y_tr, X_te, Y_te, idxs_lb, net, 
                                            train_handler, test_handler, args)    
    else:
        strategy = query_strategies.__dict__[args.strategy](X_tr, Y_tr, X_te, Y_te, idxs_lb, net, handler, args)

    print_log('Strategy {} successfully loaded...'.format(args.strategy), log)

    # load pretrained model
    if args.is_load_ckpt:
        print_log('load pretrained checkpoint and calculate the coverage, density, etc', log)
        strategy.save_metrixs(is_small_budget=True)
        return 0

    # round 0 accuracy
    alpha = 2e-3
    # if args.load_model:
    #     strategy.load_model()
    # else:
    strategy.train(alpha=alpha, n_epoch=args.n_epoch)
    test_acc= strategy.predict(X_te, Y_te)

    acc = np.zeros(NUM_ROUND+1)
    acc[0] = test_acc
    print_log('==>> Testing accuracy {}'.format(acc[0]), log)

    out_file = os.path.join(args.save_path, args.save_file)
    for rd in range(1, NUM_ROUND+1):
        print('Round {}/{}'.format(rd, NUM_ROUND), flush=True)
        labeled = len(np.arange(n_pool)[idxs_lb])
        if NUM_QUERY > int(args.nEnd*n_pool/100) - labeled:
            NUM_QUERY = int(args.nEnd*n_pool/100) - labeled
            
        # query
        ts = time.time()
        output = strategy.query(NUM_QUERY)
        q_idxs = output
        idxs_lb[q_idxs] = True
        te = time.time()
        tp = te - ts
       
        # update
        strategy.update(idxs_lb)
        best_test_acc = strategy.train(alpha=alpha, n_epoch=args.n_epoch)

        t_iter = time.time() - ts
        
        # round accuracy
        # test_acc = strategy.predict(X_te, Y_te)
        acc[rd] = best_test_acc
        print_log(str(sum(idxs_lb)) + '\t' + 'testing accuracy {}'.format(acc[rd]), log)

        print_log("logging...", log)
        with open(out_file, 'a+') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([
                            args.strategy,
                            args.rand_idx,
                            'budget',
                            args.nEnd,
                            'nStart', 
                            args.nStart,
                            'nQuery',
                            args.nQuery,
                            'labeled',
                            min(args.nStart + args.nQuery*rd, args.nEnd),
                            'accCompare',
                            acc[0],
                            acc[rd],
                            acc[rd] - acc[0],
                            't_query',
                            tp,
                            't_iter',
                            t_iter
                            ])

    print_log('success!', log)


if __name__ == '__main__':
    main()
