# This is an implementation of the paper: Sequential GCN for Active Learning

# Implemented by Yu LI, based on the code: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning

import numpy as np
from .strategy import Strategy
import pdb
from torch.nn import functional 
import math
import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from models.gcn import GCN
import torch.optim as optim
from tqdm import tqdm

EPOCH_GCN = 200
LR_GCN = 1e-3
SUBSET    = 10000 # M

def aff_to_adj(x, y=None):
    x = x.detach().cpu().numpy()
    adj = np.matmul(x, x.transpose())
    adj +=  -1.0*np.eye(adj.shape[0])
    adj_diag = np.sum(adj, axis=0) #rowise sum
    adj = np.matmul(adj, np.diag(1/adj_diag))
    adj = adj + np.eye(adj.shape[0])
    adj = torch.Tensor(adj)

    return adj


def BCEAdjLoss(scores, lbl, nlbl, l_adj):
    lnl = torch.log(scores[lbl])
    lnu = torch.log(1 - scores[nlbl])
    labeled_score = torch.mean(lnl) 
    unlabeled_score = torch.mean(lnu)
    bce_adj_loss = -labeled_score - l_adj*unlabeled_score
    return bce_adj_loss


class uncertainGCN(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(uncertainGCN, self).__init__(X, Y,  X_te, Y_te, idxs_lb, net, handler, args)
        
    def query(self, n):
        # get the features of all data (labeled + unlabeled)
        subset = list(np.nonzero(~self.idxs_lb)[0][:SUBSET])
        ind_idxs_lb = list(np.nonzero(self.idxs_lb)[0])

        features = self.get_embedding(self.X[subset+ind_idxs_lb], self.Y[subset+ind_idxs_lb])
        features = functional.normalize(features).to(self.device)
        adj = aff_to_adj(features).to(self.device)

        binary_labels = torch.cat((torch.zeros([SUBSET, 1]),(torch.ones([len(ind_idxs_lb),1]))),0)
        
        gcn_module = GCN(nfeat=features.shape[1],
                         nhid=self.args.hidden_units,
                         nclass=1,
                         dropout=self.args.dropout_rate).to(self.device)
        models      = {'gcn_module': gcn_module}

        optim_backbone = optim.Adam(models['gcn_module'].parameters(), lr=1e-3,
                                 weight_decay=5e-4)
        optimizers = {'gcn_module': optim_backbone}

        lbl = np.arange(SUBSET, SUBSET+len(ind_idxs_lb), 1) # temp labeled index
        nlbl = np.arange(0, SUBSET, 1) # temp unlabled index

        # train the gcn model
        for _ in tqdm(range(200)):
            optimizers['gcn_module'].zero_grad()
            outputs, _, _ = models['gcn_module'](features, adj)
            lamda = self.args.lambda_loss 
            loss = BCEAdjLoss(outputs, lbl, nlbl, lamda)
            loss.backward()
            optimizers['gcn_module'].step()
        
        models['gcn_module'].eval()
        with torch.no_grad():
            inputs = features.to(self.device)
            labels = binary_labels.to(self.device)
            scores, _, feat = models['gcn_module'](inputs, adj)

            s_margin = self.args.s_margin 
            scores_median = np.squeeze(torch.abs(scores[:SUBSET] - s_margin).detach().cpu().numpy())
            arg = np.argsort(-(scores_median))



        print("Max confidence value: ",torch.max(scores.data))
        print("Mean confidence value: ",torch.mean(scores.data))
        preds = torch.round(scores)
        correct_labeled = (preds[SUBSET:,0] == labels[SUBSET:,0]).sum().item() / len(ind_idxs_lb)
        correct_unlabeled = (preds[:SUBSET,0] == labels[:SUBSET,0]).sum().item() / SUBSET
        correct = (preds[:,0] == labels[:,0]).sum().item() / (SUBSET + len(ind_idxs_lb))
        print("Labeled classified: ", correct_labeled)
        print("Unlabeled classified: ", correct_unlabeled)
        print("Total classified: ", correct)
        
        subset = np.array(subset)
        inds = subset[arg][-n:]

        return inds



