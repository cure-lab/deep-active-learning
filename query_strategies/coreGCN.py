# This is an implementation of the paper: Sequential GCN for Active Learning
# Implemented by Yu LI, based on the code: https://github.com/razvancaramalau/Sequential-GCN-for-Active-Learning
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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



from sklearn.metrics import pairwise_distances
from scipy.spatial import distance

import abc


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



class SamplingMethod(object):
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def __init__(self, X, y, seed, **kwargs):
    self.X = X
    self.y = y
    self.seed = seed

  def flatten_X(self):
    shape = self.X.shape
    flat_X = self.X
    if len(shape) > 2:
      flat_X = np.reshape(self.X, (shape[0],np.product(shape[1:])))
    return flat_X


  @abc.abstractmethod
  def select_batch_(self):
    return

  def select_batch(self, **kwargs):
    return self.select_batch_(**kwargs)

  def select_batch_unc_(self, **kwargs):
      return self.select_batch_unc_(**kwargs)

  def to_dict(self):
    return None



class kCenterGreedy(SamplingMethod):

    def __init__(self, X,  metric='euclidean'):
        self.X = X
        # self.y = y
        self.flat_X = self.flatten_X()
        self.name = 'kcenter'
        self.features = self.flat_X
        self.metric = metric
        self.min_distances = None
        self.max_distances = None
        self.n_obs = self.X.shape[0]
        self.already_selected = []

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        """Update min distances given cluster centers.
        Args:
          cluster_centers: indices of cluster centers
          only_new: only calculate distance for newly selected points and update
            min_distances.
          rest_dist: whether to reset min_distances.
        """

        if reset_dist:
          self.min_distances = None
        if only_new:
          cluster_centers = [d for d in cluster_centers
                            if d not in self.already_selected]
        if cluster_centers:
          x = self.features[cluster_centers]
          # Update min_distances for all examples given new cluster center.
          dist = pairwise_distances(self.features, x, metric=self.metric)#,n_jobs=4)

          if self.min_distances is None:
            self.min_distances = np.min(dist, axis=1).reshape(-1,1)
          else:
            self.min_distances = np.minimum(self.min_distances, dist)

    def select_batch_(self, already_selected, N, **kwargs):
        """
        Diversity promoting active learning method that greedily forms a batch
        to minimize the maximum distance to a cluster center among all unlabeled
        datapoints.
        Args:
          model: model with scikit-like API with decision_function implemented
          already_selected: index of datapoints already selected
          N: batch size
        Returns:
          indices of points selected to minimize distance to cluster centers
        """

        try:
          # Assumes that the transform function takes in original data and not
          # flattened data.
          print('Getting transformed features...')
        #   self.features = model.transform(self.X)
          print('Calculating distances...')
          self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
          print('Using flat_X as features.')
          self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(N):
          if self.already_selected is None:
            # Initialize centers with a randomly selected datapoint
            ind = np.random.choice(np.arange(self.n_obs))
          else:
            ind = np.argmax(self.min_distances)
          # New examples should not be in already selected since those points
          # should have min_distance of zero to a cluster center.
          assert ind not in already_selected

          self.update_distances([ind], only_new=True, reset_dist=False)
          new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
                % max(self.min_distances))


        self.already_selected = already_selected

        return new_batch



class coreGCN(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(coreGCN, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        
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
        for _ in range(200):
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

            feat = feat.detach().cpu().numpy()
            new_av_idx = np.arange(SUBSET,(SUBSET + len(ind_idxs_lb)))
            sampling2 = kCenterGreedy(feat)  
            batch2 = sampling2.select_batch_(new_av_idx, n)
            other_idx = [x for x in range(SUBSET) if x not in batch2]
            arg = other_idx + batch2
        
        subset = np.array(subset)
        inds = subset[arg][-n:]

        print("Max confidence value: ",torch.max(scores.data))
        print("Mean confidence value: ",torch.mean(scores.data))
        preds = torch.round(scores)
        correct_labeled = (preds[SUBSET:,0] == labels[SUBSET:,0]).sum().item() / len(ind_idxs_lb)
        correct_unlabeled = (preds[:SUBSET,0] == labels[:SUBSET,0]).sum().item() / SUBSET
        correct = (preds[:,0] == labels[:,0]).sum().item() / (SUBSET + len(ind_idxs_lb))
        print("Labeled classified: ", correct_labeled)
        print("Unlabeled classified: ", correct_unlabeled)
        print("Total classified: ", correct)

        return inds



