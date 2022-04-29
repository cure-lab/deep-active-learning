from cProfile import label
from itertools import count
from tokenize import PlainToken
import numpy as np
from sklearn.cluster import KMeans
import os, sys
import matplotlib.pyplot as plt
sys.path.append("..")

import dataset

data = 'cifar10'
n_clusters = 2000

embed = np.load('../embedding/{}_embed.npy'.format(data))
cluster_learner = KMeans(n_clusters=n_clusters)
cluster_learner.fit(embed)
cluster_idxs = cluster_learner.predict(embed)


# save cluster index
# np.save('../embedding/{}_clusters.npy'.format(dataset),cluster_idxs)


# # analysis
# cluster_idxs = np.load('../embedding/{}_clusters.npy'.format(data))

# check the embedding 
# 1. check how many samples are in each cluster
unique, counts = np.unique(cluster_idxs, return_counts=True)
print (len(unique), len(counts))
plt.figure()
plt.hist(counts)
plt.savefig('samples_each_cluster_hist.png')

# 2. check how many different labels in each cluster
# load the true labels for the embedding
X_tr, Y_tr, X_te, Y_te = dataset.get_dataset(data, '/research/dept2/yuli/datasets')
num_diff_labels = []
for i in range(n_clusters):
        idxes_in_cluster = np.nonzero(cluster_idxs==i)[0]
        print ('cluter #', i)
        print ('number of samples in the cluster: ', len(idxes_in_cluster))
        labels = Y_tr[idxes_in_cluster]
        print ('unique labels in this cluster: ', np.unique(labels))
        num_unique = len(np.unique(labels))
        num_diff_labels.append(num_unique)

plt.figure()
plt.hist(num_diff_labels)
plt.savefig('unique_labels_hist.png')


