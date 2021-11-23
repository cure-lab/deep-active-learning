import numpy as np
from .strategy import Strategy
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
from sklearn.cluster import AgglomerativeClustering
distance_threshold = 8
# (ClusterMargin, 2021) Batch Active Learning at Scale
# Original code is not open source. Reproduced by muxi

class Cluster():
    def __init__(self, init_points, cluster_id):
        self.points = init_points
        self.cluster_id = cluster_id
        self.merged = False

    def inverse(self, cluster_dict):
        for point in self.points:
            cluster_dict[str(point)] = self.cluster_id
        return cluster_dict
    
    def update_id(self, cluster_id):
        self.cluster_id = cluster_id

def average_linkage(cluster_1, cluster_2):
    len_1 = len(cluster_1.points)
    len_2 = len(cluster_2.points)
    cluster_distance = 0
    for point_1 in cluster_1.points:
        for point_2 in cluster_2.points:
            cluster_distance += distance(point_1,point_2)
    cluster_distance/=(len_1*len_2)
    return cluster_distance

shortcut = {}
distance_dict = {}

def distance(point_1,point_2):
    global distance_dict, shortcut
    p1 = shortcut[str(point_1)]
    p2 = shortcut[str(point_2)]
    if p2 not in distance_dict[p1].keys():
        dist = round(np.linalg.norm(point_1 - point_2),10)
        distance_dict[p1][p2] = dist
        # distance_dict[p2][p1] = dist
    return distance_dict[p1][p2]

def HAC(points_set):
    global distance_dict, shortcut
    cluster_list = []
    for idex, point in enumerate(points_set):
        distance_dict[idex] = {}
        shortcut[str(point)] = idex
        cluster_list.append(Cluster([point,], idex))
    update_flag = True
    round = 0

    while(update_flag) and len(cluster_list) > 5000:
        new_cluster_list = []
        round += 1
        print("HAC round ", round, "Length ",len(cluster_list))
        for index_1, cluster_1 in enumerate(cluster_list):
            for index_2, cluster_2 in enumerate(cluster_list):
                if index_2 <= index_1:
                    continue
                if cluster_1.merged or cluster_2.merged:
                    continue
                avg_link = average_linkage(cluster_1,cluster_2)
                if avg_link < distance_threshold:
                    new_cluster_list.append(Cluster(cluster_1.points + cluster_2.points, cluster_1.cluster_id))
                    cluster_1.merged = True
                    cluster_2.merged = True
                # print(avg_link)
        
        for cluster in cluster_list:
            if not cluster.merged:
                new_cluster_list.append(cluster)

        if len(cluster_list) == len(new_cluster_list):
            update_flag = False
        cluster_list= new_cluster_list
        for index, cluster in enumerate(cluster_list):
            cluster.update_id(index)
            cluster.merged = False
        # print(len(new_cluster_list))
    cluster_dict = {}
    for index, cluster in enumerate(cluster_list):
        cluster_dict = cluster.inverse(cluster_dict)
        # print(cluster.points)
    return cluster_dict

class ClusterMarginSampling(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(ClusterMarginSampling, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
        self.one_sample_step = True
    
    def prepare_emb(self):
        loader_te = DataLoader(self.handler(self.X, self.Y, transform=self.args.transform_te), shuffle=False, **self.args.loader_te_args)
        self.clf.eval()
        create = True
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = Variable(x.to(self.device)), Variable(y.to(self.device))
                out, emb, feature = self.clf(x,intermediate = True) # resnet emb from last avg pool
                if create:
                    create = False
                    emb_list = torch.zeros([len(self.Y), len(feature[-1].view(out.size(0), -1)[1])])
                emb_list[idxs] = feature[-1].view(out.size(0), -1)[1].cpu().data
        return np.array(emb_list)

    def margin_data(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        # 仅获得未标注数据 prob
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled])
        probs_sorted, idxs = probs.sort(descending=True)
        U = probs_sorted[:, 0] - probs_sorted[:,1]
        # emb_list = self.emb_list
        # emb_select_list = emb_list[idxs_unlabeled[U.sort()[1].numpy()[:n]]]
        # sort函数默认 dim = -1， [1] 表示的是 index
        return idxs_unlabeled[U.sort()[1].numpy()[:n]]

    def query(self, k):
        if self.one_sample_step:
            self.one_sample_step = False
            self.emb_list = self.prepare_emb()
            # print(self.emb_list[0])
            # self.HAC_dict = HAC(self.emb_list)
            # self.HAC_list = AgglomerativeClustering(n_clusters=None,distance_threshold = distance_threshold,linkage = 'average').fit(self.emb_list)
            # 全部数据 建立 cluster
            self.HAC_list = AgglomerativeClustering(n_clusters=20, linkage = 'average').fit(self.emb_list)

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        n = min(k*10,len(self.Y[idxs_unlabeled]))
        index = self.margin_data(n)
        index = self.round_robin(index, self.HAC_list.labels_, k)
        # print(len(index),len([i for i in index if i in idxs_unlabeled]))
        return index

    def round_robin(self, unlabeled_index, hac_list, k):
        cluster_list = []
        # print("Round Robin")
        for i in range(len(self.Y)):
            cluster = []
            cluster_list.append(cluster)
        for real_idx in unlabeled_index:
            i = hac_list[real_idx]
            cluster_list[i].append(real_idx)
        cluster_list.sort(key=lambda x:len(x))
        index_select = []
        cluster_index = 0
        # print("Select cluster",len(set(hac_list)))
        while k > 0:
            if len(cluster_list[cluster_index]) > 0:
                index_select.append(cluster_list[cluster_index].pop(0)) 
                k -= 1
            if cluster_index < len(cluster_list) - 1:
                cluster_index += 1
            else:
                cluster_index = 0

        return index_select
