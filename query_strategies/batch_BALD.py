import numpy as np
from .strategy import Strategy
import dataclasses
import typing
from batchbald_redux.batchbald import get_batchbald_batch # pip install batchbald_redux
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
""""
因为API使用了限制内存的方法toma，导致速度巨慢，可以尝试querry小batch（500张左右，gpu上不到一分钟）但不进行内存限制
"""

@dataclasses.dataclass
class AcquisitionBatch:
    indices: typing.List[int]
    scores: typing.List[float]
    orignal_scores: typing.Optional[typing.List[float]]

class BatchBALD(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BatchBALD_reduce, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.net = net
        self.args = args

    def compute_NKC(self, X,Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        K = 10 # MC采样
        self.clf.train()
        probs = torch.zeros([K, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(K):
                for x, y, idxs in loader_te:
                    x, y = Variable(x.cuda()), Variable(y.cuda())
                    out, e1 = self.clf(x)
                   
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
                    
            return probs.permute(1,0,2)
        
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        prob_NKC = self.compute_NKC(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        with torch.no_grad():
            batch = get_batchbald_batch(prob_NKC, n, 10000000) # 第三个参数不确定
        return idxs_unlabeled[batch.indices]
