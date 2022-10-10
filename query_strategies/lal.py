
import numpy as np
from .strategy import Strategy
import pdb
import math
from sklearn.ensemble import RandomForestRegressor
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
from torch.utils.data import DataLoader
from torch import nn
import torch.optim as optim
import torch
import time

# Implementation of the paper: 
# Code for paper Ksenia Konyushkova, Raphael Sznitman, Pascal Fua 'Learning Active Learning from Data', NIPS 2017
# Based on https://github.com/ksenia-konyushkova/LAL
# Note: this method is only suitable for binary classification problem and random forest model
#  and hence not used in our experiment

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
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        super(LearningAL, self).__init__(X, Y,  X_te, Y_te, idxs_lb, net, handler, args)
        self.n_estimators = args.n_estimators
        self.X_te = X_te
        self.Y_te = Y_te
        all_data_for_lal, all_labels_for_lal = self.get_features(net, )
        self.lalModel = self.train_lal_model(self, all_data_for_lal, all_labels_for_lal)
    
    def getLALdatapoints(self, n_points_per_experiment):
        # train the model based on the labeled data
        self.train(n_epoch=self.args.n_epoch)
        nFeatures = 8
        # get my features
        # now we need the number of training datapoints as a feature
        known_data = np.arange(self.n_pool)[self.idxs_lb]
        unknown_data = np.arange(self.n_pool)[~self.idxs_lb]
        known_labels = self.Y[np.arange(self.n_pool)[self.idxs_lb]]
        unknown_labels = self.Y[np.arange(self.n_pool)[~self.idxs_lb]]
        feature_vector = self._getFeaturevector4LAL(self.clf, unknown_data[0:n_points_per_experiment,:], known_labels, nFeatures)
        
        # predict on test data to evaluate the classifier quality
        test_0_acc = self.predict(self.X_te, self.Y_te)  
        # sample n_points_per_experiment samples that we will add to the training dataset and check the change in error
        gains_quality = np.zeros((n_points_per_experiment))
        
        for i in range(n_points_per_experiment):
            # try to add it to the labelled data
            new_known_data = np.concatenate((known_data,[unknown_data[i,:]]))
            new_known_labels = np.concatenate((known_labels,unknown_labels[i]))

            # train updated model - model_i
            new_known_labels = np.ravel(new_known_labels)
            self.train(self.args.n_epoch, self.X[new_known_data], self.Y[new_known_labels])
                    
            # predict on test data
            test_i_acc = self.clf.predict(self.X_te, self.Y_te)   
            # how much the quality has changed
            gains_quality[i]=(test_i_acc - test_0_acc)


        return feature_vector, gains_quality    


    def _getFeaturevector4LAL(self, unknown_data, known_labels, nFeatures):
        
        # - predicted mean (but only for n_points_per_experiment datapoints)
        prediction_unknown = self.clf.predict_proba(unknown_data)
        
        # features are in the following order:
        # 1: prediction probability
        # 2: prediction variance
        # 3: proportion of positive class
        # 4: oob score
        # 5: coeficiant of variance of feature importance
        # 6: variance of forest
        # 7: average depth of trees
        # 8: number of datapoints in training
        
        f_1 = prediction_unknown[:,0]
        # - predicted standard deviation 
        # need to call each tree of a forest separately to get a prediction because it is not possible to get them all immediately
        # f_2 = np.std(np.array([tree.predict_proba(unknown_data)[:,0] for tree in model.estimators_]), axis=0)
        # - proportion of positive points
        # check np.size(self.indecesKnown)
        f_3 = (sum(known_labels>0)/np.size(self.indecesKnown))*np.ones_like(f_1)
        # the score estimated on out of bag estimate
        # f_4 = model.oob_score_*np.ones_like(f_1)
        # - coeficient of variance of feature importance
        # check if this is the number of features!
        # f_5 = np.std(model.feature_importances_/self.dataset.trainData.shape[1])*np.ones_like(f_1)
        # - estimate variance of forest by looking at avergae of variance of some predictions
        # f_6 = np.mean(np.std(np.array([tree.predict_proba(unknown_data)[:,0] for tree in model.estimators_]), axis=0))*np.ones_like(f_1)
        # - compute the average depth of the trees in the forest
        # f_7 = np.mean(np.array([tree.tree_.max_depth for tree in model.estimators_]))*np.ones_like(f_1)            
        # LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7]), axis=0)
        LALfeatures = np.concatenate(([f_1], [f_3]), axis=0)

        if nFeatures>7:
            # the same as f_3, check np.size(self.indecesKnown)
            f_8 = np.size(self.indecesKnown)*np.ones_like(f_1)
            # LALfeatures = np.concatenate(([f_1], [f_2], [f_3], [f_4], [f_5], [f_6], [f_7], [f_8]), axis=0)
            LALfeatures = np.concatenate(([f_1], [f_3]), axis=0)

        LALfeatures = np.transpose(LALfeatures)        
        
        return LALfeatures


    def train_lal_model(self, all_data_for_lal, all_labels_for_lal):
        # build the lal model

        lalModel = self.train_lal_model(all_data_for_lal, all_labels_for_lal)

        parameters = {'est': 2000, 'depth': 40, 'feat': 6 }
        # the regression model to predict the error of an unlabeled image
        lal_model = LALmodel(all_data_for_lal, all_labels_for_lal)
        
        lal_model.builtModel(est=parameters['est'], 
                             depth=parameters['depth'], 
                             feat=parameters['feat'])
        lal_model.crossValidateLALmodel()
        print('Train Regressor Done!')
        print('Oob score = ', lal_model.model.oob_score_)
        return lal_model.model


    def query(self, n):
        # features are in the following order:
        # 1: prediction probability
        # 2: prediction variance
        # 3: proportion of positive class
        # 4: oob score
        # 5: coeficiant of variance of feature importance
        # 6: variance of forest
        # 7: average depth of trees
        # 8: number of datapoints in training
        

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

    def train(self, n_epoch=10, X=None, Y=None):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                m.reset_parameters()
        
        self.clf =  self.clf.apply(weight_reset)
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr,
                         weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        previous_loss = 0.
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr

            if X is None and Y is None:
                X_train = self.X[idxs_train]
                Y_train = torch.Tensor(self.Y.numpy()[idxs_train]).long()
            else:
                X_train = X
                Y_train = Y

            loader_tr = DataLoader(self.handler(X_train, Y_train,
                                    transform=transform), shuffle=True, 
                                    **self.args.loader_tr_args)
             
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr, optimizer)

                # measure elapsed time
                epoch_time.update(time.time() - ts)

                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Train Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(True),
                                                               1. - recorder.max_accuracy(True)))
                
                
                recorder.update(epoch, train_los, train_acc, 0, 0)

                # The converge condition 
                if abs(previous_loss - train_los) < 0.0001:
                    break
                else:
                    previous_loss = train_los

            self.clf = self.clf.module
        best_train_acc = recorder.max_accuracy(istrain=True)
        return best_train_acc                
