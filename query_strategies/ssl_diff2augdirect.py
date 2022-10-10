import numpy as np
from .semi_strategy import semi_Strategy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable

class TransformTwice:
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2

class ssl_Diff2AugDirect(semi_Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ssl_Diff2AugDirect, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)

    def predict_prob_aug(self, X, Y):
        
        loader_te = DataLoader(self.handler(X, Y, transform=TransformTwice(self.args.transform_te, self.args.transform_tr)), shuffle=False, **self.args.loader_te_args)

        self.ema_model.eval()

        probs = torch.zeros([len(Y), len(np.unique(self.Y))]).to(self.device)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x1, x2, y = x[0].to(self.device), x[1].to(self.device), y.to(self.device)
                out1, e1 = self.ema_model(x1)
                out2, e1 = self.ema_model(x2)
                probs[idxs] = (torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)) / 2
        
        return probs.cpu()

    def margin_data(self,k):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_aug(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1].numpy()[:k]]

    def query(self, k):
        index = self.margin_data(k)
        return index

