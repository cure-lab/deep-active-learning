
import numpy as np
from .strategy import Strategy
import pdb
import math

# Implementation of the paper: 

class ActiveLearnerLAL(Strategy):
    '''Points are sampled according to a method described in K. Konyushkova, R. Sznitman, P. Fua 'Learning Active Learning from data'  '''
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(ActiveLearnerLAL, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.model = RandomForestClassifier(self.nEstimators, oob_score=True, n_jobs=8)
    
    def selectNext(self):
        
        unknown_data = self.dataset.trainData[self.indicesUnknown,:]
        known_labels = self.dataset.trainLabels[self.indicesKnown,:]
        n_lablled = np.size(self.indicesKnown)
        n_dim = np.shape(self.dataset.trainData)[1]
        
        # predictions of the trees
        temp = np.array([tree.predict_proba(unknown_data)[:,0] for tree in self.model.estimators_])
        # - average and standard deviation of the predicted scores
        f_1 = np.mean(temp, axis=0)
        f_2 = np.std(temp, axis=0)
        # - proportion of positive points
        f_3 = (sum(known_labels>0)/n_lablled)*np.ones_like(f_1)
        # the score estimated on out of bag estimate
        f_4 = self.model.oob_score_*np.ones_like(f_1)
        # - coeficient of variance of feature importance
        f_5 = np.std(self.model.feature_importances_/n_dim)*np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        f_6 = np.mean(f_2, axis=0)*np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        f_7 = np.mean(np.array([tree.tree_.max_depth for tree in self.model.estimators_]))*np.ones_like(f_1)
        # - number of already labelled datapoints
        f_8 = np.size(self.indicesKnown)*np.ones_like(f_1)
        
        # all the featrues put together for regressor
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        LALfeatures = np.transpose(LALfeatures)
            
        # predict the expercted reduction in the error by adding the point
        LALprediction = self.lalModel.predict(LALfeatures)
        # select the datapoint with the biggest reduction in the error
        selectedIndex1toN = np.argmax(LALprediction)
        # retrieve the real index of the selected datapoint    
        selectedIndex = self.indicesUnknown[selectedIndex1toN]
            
        self.indicesKnown = np.concatenate(([self.indicesKnown, np.array([selectedIndex])]))
        self.indicesUnknown = np.delete(self.indicesUnknown, selectedIndex1toN)  