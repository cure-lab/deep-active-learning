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
BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning
Reproduce through API
"""

@dataclasses.dataclass
class AcquisitionBatch:
    indices: typing.List[int]
    scores: typing.List[float]
    orignal_scores: typing.Optional[typing.List[float]]

class BatchBALD(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(BatchBALD, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.net = net
        self.args = args

    def compute_NKC(self, X,Y):
        loader_te = DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            shuffle=False, **self.args['loader_te_args'])
        K = 10 # MC
        self.clf.train()
        probs = torch.zeros([K, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(K):
                for x, y, idxs in loader_te:
                    x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                    out, e1 = self.clf(x)
                   
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
                    
            return probs.permute(1,0,2)
        
    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        prob_NKC = self.compute_NKC(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])
        with torch.no_grad():
            batch = get_batchbald_batch(prob_NKC, n, 10000000) # Don't know the meaning of the third argument
        return idxs_unlabeled[batch.indices]
