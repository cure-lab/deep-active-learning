import numpy as np
from .semi_strategy import semi_Strategy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from sklearn.cluster import KMeans

class TransformTwice:
    def __init__(self, transform1,transform2):
        self.transform1 = transform1
        self.transform2 = transform2

    def __call__(self, inp):
        out1 = self.transform1(inp)
        out2 = self.transform2(inp)
        return out1, out2

class ssl_Diff2AugKmeans(semi_Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ssl_Diff2AugKmeans, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
    
    def prepare_emb(self):
        loader_te = DataLoader(self.handler(self.X, self.Y, transform=TransformTwice(self.args.transform_te, self.args.transform_tr )), shuffle=False, **self.args.loader_te_args)
        self.ema_model.eval()
        create = True
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x[0].to(self.device)), Variable(y.to(self.device))
                out, emb, feature = self.ema_model(x,intermediate = True) # resnet emb from last avg pool
                if create:
                    create = False
                    emb_list = torch.zeros([len(self.Y), len(feature[-1].view(out.size(0), -1)[1])])
                emb_list[idxs] = feature[-1].view(out.size(0), -1)[1].cpu().data
        return np.array(emb_list)

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

    def margin_data(self):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        probs = self.predict_prob_aug(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        return idxs_unlabeled[U.sort()[1].numpy()]

    def query(self, k):
        self.emb_list = self.prepare_emb()
        self.Kmeans_list = KMeans(n_clusters=20).fit(self.emb_list)
        margin_sorted_index = self.margin_data()
        index = self.diff2_aug_kmeans(margin_sorted_index, self.Kmeans_list.labels_, k)

        return index

    def diff2_aug_kmeans(self, unlabeled_index, Kmeans_list, k):
        cluster_list = []
        for i in range(20):
            cluster = []
            cluster_list.append(cluster)

        # unlabeled_index is sorted by margin score, pop(0) is the most uncertain one

        for real_idx in unlabeled_index:
            cluster_list[Kmeans_list[real_idx]].append(real_idx)

        index_select = []
        total_selected = k

        for cluster_index in range(len(cluster_list)):
            num_select_by_propotion = total_selected * len(cluster_list[cluster_index]) / len(unlabeled_index)
            for i in range(int(num_select_by_propotion)):
                index_select.append(cluster_list[cluster_index].pop(0)) 
                k -= 1

        cluster_index = 0

        # int后可能不足k个，补足
        while k > 0:
            if len(cluster_list[cluster_index]) > 0:
                index_select.append(cluster_list[cluster_index].pop(0)) 
                k -= 1
            if cluster_index < len(cluster_list) - 1:
                cluster_index += 1
            else:
                cluster_index = 0

        return index_select
