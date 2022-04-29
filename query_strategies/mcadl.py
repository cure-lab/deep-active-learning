import numpy as np
from .strategy import Strategy
import pdb
import heapq
import copy
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import pairwise

# An implementation of the paper: 
# Multi-criteria active deep learning for image classification
# Published in Knowledge-Based System, 2019
# Hunan University
def _resize(X,length):
    # Smooth augmentation, for robust performance
    import PIL
    # print(X.shape)
    X = PIL.Image.fromarray(X)
    X = X.resize((length,length))
    return np.array(X)

class MCADL(Strategy):
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(MCADL, self).__init__(X, Y,  X_te, Y_te, idxs_lb, net, handler, args)
        self.alpha_init = 0.9
        self.beta_init = 0.9
        self.last_acc = [0.0]*self.args.n_class
        X_sim = []
        # print(type(self.X_te[0]))
        # exit()
        for x in X:
            X_sim.append(_resize(x,self.args.img_size))
        X_sim = np.array(X_sim)
        self.similarity = pairwise.cosine_similarity(X_sim.reshape([len(X), -1]), X_sim.reshape([len(X), -1]))

    def uncertainty(self, proba, flag):
        '''
        Input:
        @proba: probability for all samples, n_sample x nb_class
        @flag: a mark for samples, n_sample x 1, 0 represents unselected
        return:
        @BvSB: the uncertainty measure
        @pse_index: the index of sample to be psesudo labeled
        '''
        n = proba.shape[0]
        P_index = np.where(np.array(flag)==0) #1xN array
        plist = P_index[0]
        uncert  = -1.0*np.ones((n,),dtype='float64')
        for d in plist:
            D = proba[d,:]
            # decide k
            k = 0
            while True:
                k += 1
                lgst = heapq.nlargest(k, D)
                if sum(lgst) > 0.5:
                    break

            Z = np.array(heapq.nlargest(k, D))
            v = np.absolute(Z - Z.mean()).mean()
            uncert[d] = 1-v

        return uncert

    def evaluate_each_class(self, X, Y):
        '''
        Input:
        @X: data
        @Y: label
        Return:
        @Acc: the accuracy of all classes
        '''
        Acc = []
        for i in range(self.args.n_class):
            # obtain the accuracy on each class
            if len(Y[Y==i]) == 0:
                acc = 0
            else:
                acc = self.predict(X[Y==i], Y[Y==i])
            Acc.append(acc)
        return Acc


    def getID(self, pred_label, listID):
        '''
        Input:
        @pred: the predicted label for the all samples
        @listID: the index of k neighbor to current sample
        Return:
        @maxID: the psesudo label
        The psesudo label is decided by the surrounding labels
        '''
        pclass=[]
        for i in listID:
            pclass.append(pred_label[i])
        setClass = set(pclass)
        maxID = -1
        maxCount =-1
        for i in setClass:
            if pclass.count(i)>maxCount:
                maxID =i
                maxCount = pclass.count(i)
    #    print("knn class:",pclass,"final ",maxID)
        return maxID

    def avg_cosine_distance(self, idx, listed):
        '''
        Input:
        @idx: the index of the sample to be measured density
        @listed: the index of the labeled samples
        Return:
        @The average value of diversity
        @m: the index of k neighbor to the sample to be measured density
        '''

        n = len(listed)
        sumD = sum(1-self.similarity[idx, listed])

        #modified 0805
        list_value = list(self.similarity[idx, listed])
        kvalue = heapq.nlargest(5, list_value)
        m = []
        for i in range(len(kvalue)):
            ids = list_value.index(kvalue[i])
            v = listed[ids]
            m.append(v)
        return 1.0*sumD/n, m

    def getWeightM(self, last_acc, now_acc):
        '''
        Input:
        @last_acc: the accuracy of last round
        @now_acc: the accuracy of current round
        Return:
        @q: the weights of all classes
        '''
        b = 0.5
        q=[]
        # in the begining, the approach pays more attention to 
        # samples from classes that have fast performance enhancement
        if min(now_acc) < b:
            p=copy.deepcopy(now_acc)
            s=0
            print("first ....")
            for i in range(self.args.n_class):
                t=p[i]-last_acc[i]
                if t<0:
                    t=0
                s+=t
                q.append(t)
            print("improvement:", q)
            if s!=0.0:
                for i in range(self.args.n_class):
                    q[i]/=s

            else:
                q=[0.1]*10
        else:
            # as the performance continues to improve, 
            # they tends to select samples from the clases with low performance 
            # to balance the performance among classes
            q = [1./i for i in list(now_acc)]
            q = [i/sum(q) for i in q]

        print("last_acc:", last_acc)
        print("acc:", now_acc)
        print("final weight:", q)
        return q

    def query(self, n):
        class_accs = self.evaluate_each_class(self.X[self.idxs_lb], self.Y[self.idxs_lb])
        weight_classes = self.getWeightM(self.last_acc, class_accs)
        self.last_acc = class_accs

        # calculate alpha and beta
        AR_t = self.predict(self.X[self.idxs_lb], self.Y[self.idxs_lb]) # average acc on training data
        alpha = self.alpha_init * np.exp(-AR_t)
        beta = self.beta_init * np.exp(-AR_t)

        proba = self.predict_prob(self.X, self.Y)
        
        idxs_unlabeled = np.nonzero(~self.idxs_lb)[0]
        idxs_labeled = np.nonzero(self.idxs_lb)[0]
        # for each sample, calculate its informativeness
        # density and similarity
        density = [-1.0]*len(self.idxs_lb)
        similarity = [-1.0]*len(self.idxs_lb)
        for i in list(idxs_unlabeled):
            # for an unlabeled sample x_i, we determine its most similar samples x_s
            distance, kId = self.avg_cosine_distance(i, idxs_labeled)
            # The pesudo label of x_i is deciding by x_s
            pse_class = self.getID(proba.max(1)[1], kId)
            # cosine distance to the labeled samples belongs to the pesudo label
            sample_idxs = idxs_labeled[self.Y[self.idxs_lb] == pse_class]
            cosine_distance = self.similarity[i, sample_idxs]

            density[i] = 1 - cosine_distance.mean()
            similarity[i] = 1 - cosine_distance.max()

        
        # uncertainty of all samples
        uncertainty_score = self.uncertainty(proba=proba, flag=self.idxs_lb)
        
        # label-based measure
        label_based_score = [-1.0]*len(self.idxs_lb)

        for i in list(idxs_unlabeled):
            # for an unlabeled sample x_i, we determine its most similar samples x_s
            distance, kId = self.avg_cosine_distance(i, idxs_labeled)
            # The pesudo label of x_i is deciding by x_s
            pse_class = self.getID(proba.max(1)[1], kId)
            # finally, the info-label value of x_i is equal to w_s
            w_classes = weight_classes[pse_class]
            # score[i] = BvSB[i,]
            label_based_score[i] = w_classes
        

        info_data = [0.5*density[i] + 0.5*similarity[i] for i in range(len(density))]
        info_model = [beta*uncertainty_score[i] + (1-beta)*label_based_score[i] for i in range(len(label_based_score))]
        infomativeness = [alpha*info_data[i] + (1-alpha)*info_model[i] for i in range(len(info_model))]
        infomativeness = np.array(infomativeness)[idxs_unlabeled]

        # select the top-n samples
        ranked_idx = infomativeness.argsort()[::-1][:n]

        return idxs_unlabeled[ranked_idx]

