import numpy as np
from .semi_strategy import semi_Strategy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

class TransformFive:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        transform_list = []
        for i in range(50):
            transform_list.append(self.transform(inp))
        return transform_list

class TransformFifty:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, inp):
        transform_list = []
        for i in range(50):
            transform_list.append(self.transform(inp))
        return transform_list

class ssl_Consistency(semi_Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ssl_Consistency, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def predict_consistency_aug(self, X, Y):
        
        loader_consistency = DataLoader(self.handler(X, Y, transform=TransformFive(self.args.transform_tr)), shuffle=False, **self.args.loader_te_args)

        self.ema_model.eval()

        consistency = np.zeros([len(Y)])
        with torch.no_grad():
            for x, y, idxs in loader_consistency:
                probs = np.zeros([len(x), y.size(0),len(np.unique(self.Y))])
                for i, xi in enumerate(x):
                    out1, e1 = self.ema_model(xi.to(self.device))
                    prob = torch.softmax(out1, dim=1).cpu()
                    probs[i] = prob

                consistency[idxs] = probs.var(0).sum(1)
        
        return consistency

    def query(self,k):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # print(print(len(set(idxs_unlabeled))), idxs_unlabeled)
        consistency = self.predict_consistency_aug(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        idxs = consistency.argsort() 
        # print(idxs)
        # print(len(set(idxs[:k])),len(idxs[:k]),len(set(idxs_unlabeled)), idxs_unlabeled[idxs[:k]])
        return idxs_unlabeled[idxs[-k:]]


