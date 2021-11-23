import numpy as np
from .semi_strategy import semi_Strategy

class ssl_Random(semi_Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ssl_Random, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def query(self, n):
        inds = np.where(self.idxs_lb==0)[0]
        return inds[np.random.permutation(len(inds))][:n]
