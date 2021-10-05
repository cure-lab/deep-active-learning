
import numpy as np
from .strategy import Strategy
import pdb
import math
from sklearn.ensemble import RandomForestRegressor

# Implementation of the paper: 
# Code for paper Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active Learning from Data', NIPS 2017
# Based on https://github.com/ksenia-konyushkova/LAL

class LALmodel:
    ''' Class for the regressor that predicts the expected error reduction caused by adding datapoints'''
    
    def __init__(self, all_data_for_lal, all_labels_for_lal):
        
        self.all_data_for_lal = all_data_for_lal
        self.all_labels_for_lal = all_labels_for_lal
        
    def crossValidateLALmodel(self, possible_estimators, possible_depth, possible_features):
        ''' Cross-validate the regressor model.
        input: possible_estimators -- list of possible number of estimators (trees) in Random Forest regression
        possible_depth -- list of possible maximum depth of the tree in RF regressor
        possible_features -- list of possible maximum number of features in a split of tree in RF regressor'''
            
        best_score = -math.inf

        self.best_est = 0
        self.best_depth = 0
        self.best_feat = 0
    
        print('start cross-validating..')
        for est in possible_estimators:
            for depth in possible_depth:
                for feat in possible_features:
                    model = RandomForestRegressor(n_estimators = est, max_depth=depth, max_features=feat, oob_score=True, n_jobs=8)
                    model.fit(self.all_data_for_lal[:,:], np.ravel(self.all_labels_for_lal))
                    if model.oob_score_>best_score:
                        self.best_est = est
                        self.best_depth = depth
                        self.best_feat = feat
                        self.model = model
                        best_score = model.oob_score_
                    print('parameters tested = ', est, ', ', depth, ', ', feat, ', with the score = ', model.oob_score_)
        # now train with the best parameters
        print('best parameters = ', self.best_est, ', ', self.best_depth, ', ', self.best_feat, ', with the best score = ', best_score)
        return best_score
    
    
    def builtModel(self, est, depth, feat):
        ''' Fits the regressor with the parameters identifier as an input '''
            
        self.model = RandomForestRegressor(n_estimators = est, max_depth=depth, max_features=feat, oob_score=True, n_jobs=8)
        self.model.fit(self.all_data_for_lal, np.ravel(self.all_labels_for_lal))
        print('oob score = ', self.model.oob_score_)


class LearningAL(Strategy):
    '''Points are sampled according to a method described in K. Konyushkova, R. Sznitman, P. Fua 'Learning Active Learning from data'  '''
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(LearningAL, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.n_estimators = args.n_estimators

    def train_lal_model(self, all_data_for_lal, all_labels_for_lal):
        # the regression model to predict the error of an unlabeled image
        lal_model = LALmodel(all_data_for_lal, all_labels_for_lal)

        

        return lal_model

    def query(self, n):
        # prepare the lal model
        all_data_for_lal = 
        all_labels_for_lal = 
        lalModel = self.train_lal_model(all_data_for_lal, all_labels_for_lal)

        # data
        known_labels = self.Y[self.idxs_lb]

        n_lablled = np.sum(self.idxs_lb)
        n_dim = np.shape(self.X)[1]
        
        # predict probabilities for the unlabeled data
        temp = self.predict_prob(self.X[~self.idxs_lb], self.Y[~self.idxs_lb])

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
        f_8 = n_lablled*np.ones_like(f_1)
        
        # all the featrues put together for regressor
        LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
        LALfeatures = np.transpose(LALfeatures)
            
        # predict the expercted reduction in the error by adding the point
        LALprediction = self.lalModel.predict(LALfeatures)

        # select the datapoint with the biggest reduction in the error
        selectedIndex1toN = np.argsort(LALprediction)[::-1][:n] 

        # retrieve the real index of the selected datapoint
        indicesUnknown = np.nonzero(~self.idxs_lb)[0]    
        selectedIndex = indicesUnknown[selectedIndex1toN]
            
        return selectedIndex