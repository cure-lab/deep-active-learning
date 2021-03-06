from joblib.externals.cloudpickle.cloudpickle import instance
import numpy as np
import random
from sklearn import preprocessing
from torch import nn
import sys, os
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import time
from torchvision.utils import save_image
from tqdm import tqdm
from .util import AugMixDataset
from sklearn.metrics import pairwise_distances
class Strategy:
    def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args):
        self.X = X  # vector
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te

        self.idxs_lb = idxs_lb # bool type
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.pretrained = args.pretrained
        if args.pretrained:
            self.preprocessing = args.preprocessing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.device = torch.device('cuda', args.d) if torch.cuda.is_available() else 'cpu'
        if isinstance(net,list):
            self.clf = [net_.to(self.device) for net_ in net]
            self.net = [net_.to(self.device) for net_ in net]
        else:
            self.clf = net.to(self.device)
            self.net = net.to(self.device)

        if self.pretrained: # use the latent vector of the inputs as training data
            self.X_p = self.get_pretrained_embedding(X, Y)
        
        # for reproducibility
        self.g = torch.Generator()
        self.g.manual_seed(0)

    def seed_worker(self, worker_id):
        """
        To preserve reproducibility when num_workers > 1
        """
        # https://pytorch.org/docs/stable/notes/randomness.html
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_tr, optimizer):
        self.clf.train()

        accFinal = 0.
        train_loss = 0.
        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            x, y = x.to(self.device), y.to(self.device) 
            nan_mask = torch.isnan(x)
            if nan_mask.any():
                raise RuntimeError(f"Found NAN in input indices: ", nan_mask.nonzero())

            # exit()
            optimizer.zero_grad()

            out, e1 = self.clf(x) if not self.pretrained else self.clf.module.classifier(x)
            nan_mask_out = torch.isnan(y)
            if nan_mask_out.any():
                raise RuntimeError(f"Found NAN in output indices: ", nan_mask.nonzero())
                
            loss = F.cross_entropy(out, y)

            train_loss += loss.item()
            accFinal += torch.sum((torch.max(out,1)[1] == y).float()).data.item()
            
            loss.backward()
            
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            optimizer.step()
            
            if batch_idx % 10 == 0:
                print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / len(loader_tr.dataset.X), train_loss

    
    def train(self, alpha=0.1, n_epoch=10):
        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                m.reset_parameters()
        
        self.clf =  self.clf.apply(weight_reset)
        # if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # self.clf = nn.parallel.DistributedDataParallel(self.clf,
                                                        # find_unused_parameters=True,
                                                        # )
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters() if not self.pretrained else self.clf.module.classifier.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.

        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr if not self.pretrained else None

            train_data = self.handler(self.X[idxs_train] if not self.pretrained else self.X_p[idxs_train], 
                                torch.Tensor(self.Y[idxs_train]).long() if type(self.Y) is np.ndarray else  torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform)

            loader_tr = DataLoader(train_data, 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args)
            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr, optimizer)
                test_acc = self.predict(self.X_te, self.Y_te)
                # measure elapsed time
                epoch_time.update(time.time() - ts)
                print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Test Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               1. - recorder.max_accuracy(False)))
                recorder.update(epoch, train_los, train_acc, 0, test_acc)

            if self.args.save_model:
                self.save_model()
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module

        if self.args.save_tta:
            test_acc = self.predict(self.X_te, self.Y_te)
            self.save_tta_values(self.get_tta_values(),train_los,epoch, train_acc, test_acc)

        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc                


    def predict(self, X, Y):
        # add support for pretrained model
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)
        
        if not self.pretrained:
            self.clf.eval()
        else:
            self.clf.classifier.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                pred = out.max(1)[1]                
                correct +=  (y == pred).sum().item() 

            test_acc = 1. * correct / len(Y)
   
        return test_acc

    def get_prediction(self, X, Y):
        # add support for pretrained model
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)

        P = torch.zeros(len(X)).long().to(self.device)

        if not self.pretrained:
            self.clf.eval()
        else:
            self.clf.classifier.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                pred = out.max(1)[1]     
                P[idxs] = pred           
                correct +=  (y == pred).sum().item() 
   
        return P

    def predict_prob(self, X, Y):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, pin_memory=True, **self.args.loader_te_args)

        if not self.pretrained:
            self.clf.eval()
        else:
            self.clf.classifier.eval()

        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)
        if not self.pretrained:
            self.clf.train()
        else:
            self.clf.classifier.train()

        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.clf(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)

        if not self.pretrained:
            self.clf.train()
        else:
            self.clf.classifier.train()

        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.clf(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def get_embedding(self, X, Y):
        """ get last layer embedding from current model"""
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)
        if not self.pretrained:
            self.clf.eval()
        else:
            self.clf.classifier.eval()
        
        embedding = torch.zeros([len(Y), 
                self.clf.module.get_embedding_dim() if isinstance(self.clf, nn.DataParallel) 
                else self.clf.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                embedding[idxs] = e1.data.cpu().float()
        
        return embedding

    def get_pretrained_embedding(self, X, Y):
        """ get embedding from the pretrained model: add by Yuli
            Only valid if pretrained is true
        """
        if not self.pretrained:
            raise ValueError("pretrained is not true")

        transform = self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)
    
        embedding = torch.zeros([len(Y), 512])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                e1 = self.clf.fe.encode_image(x)
                embedding[idxs] = e1.data.cpu().float()
        
        return embedding
    
    def get_grad_embedding(self, X, Y):
        """ gradient embedding (assumes cross-entropy loss) of the last layer"""
        transform = self.args.transform_te if not self.pretrained else self.preprocessing

        model = self.clf
        if isinstance(model, nn.DataParallel):
            model = model.module
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True,
                            shuffle=False, **self.args.loader_te_args)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                cout, out = self.clf(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                for j in range(len(y)):
                    for c in range(nLab):
                        if c == maxInds[j]:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        else:
                            embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
    
    def save_model(self):
        # save model and selected index
        save_path = os.path.join(self.args.save_path,self.args.dataset)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(float(100*labeled/len(self.X)))
        torch.save(self.clf, os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.manualSeed)+'.pkl'))
        print('save to ',os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.manualSeed)+'_parameter.pkl'))
        path = os.path.join(save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.manualSeed)+'.npy')
        np.save(path,self.idxs_lb)

    def load_model(self, is_small_budget=False):
        if is_small_budget:
            model_path = os.path.join('checkpoint', self.args.dataset + '_smallbudget')
        else:
            model_path = os.path.join('checkpoint', self.args.dataset + '_' + self.args.strategy)
        self.clf = torch.load(os.path.join(model_path, self.args.strategy+'_'+self.args.model+'_'+self.args.load_model+'_'+str(self.args.manualSeed)+'.pkl'))
        self.idxs_lb = np.load(os.path.join(model_path, self.args.strategy+'_'+self.args.model+'_'+self.args.load_model+'_'+str(self.args.manualSeed)+'.npy'))

    def get_tta_values(self,save_info=False):    
        # Save Energy, Energy Variance, Smoothness, Entropy for analysis 
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(self.X_te, self.Y_te, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)
        self.clf.eval()
        origin_Energy = torch.zeros(len(self.Y_te))
        origin_Entropy = torch.zeros(len(self.Y_te))
        origin_Confidence = torch.zeros(len(self.Y_te))
        origin_Margin = torch.zeros(len(self.Y_te))
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.clf(x)
                prob = F.softmax(out, dim=1)
                log_probs = torch.log(prob)
                origin_Entropy[idxs] = (-prob*log_probs).cpu().sum(1)
                origin_Confidence[idxs] = prob.max(1)[0].cpu()
                probs_sorted, _ = prob.sort(descending=True)
                origin_Margin[idxs] = probs_sorted[:, 0].cpu() - probs_sorted[:,1].cpu()
                origin_Energy[idxs] = -np.log(np.exp(out.data.cpu()).sum(1))
        if save_info:
            labeled = len(np.arange(self.n_pool)[self.idxs_lb])
            labeled_percentage = str(int(100*labeled/len(self.X)))
            np.save(os.path.join(self.args.save_path,'ECM_%s_%s.npy'%(self.args.strategy,labeled_percentage)), np.array([origin_Entropy,origin_Confidence,origin_Margin]))

        origin_Energy = origin_Energy.numpy()
        n_class, n_iters = self.args.n_class, 32 
        batch_size = 100
        n_image = n_iters + 1
        tta_values = []
        entropy_values = []
        energy_Var = []
        energyVar_values = []
        augset = AugMixDataset(self.X_te, self.args.transform_te, n_iters) 
        augtest_loader = torch.utils.data.DataLoader(
                        augset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=self.args.loader_te_args['num_workers'],
                        pin_memory=True)
        self.clf.eval()
        new_idx = []
        for k in range(batch_size):
            for j in range(n_image):
                new_idx.append(k+j*batch_size)
        for i, (all_images,idx) in tqdm(enumerate(augtest_loader)):
            all_logits, _ = self.clf(torch.cat(all_images, 0).to(self.device))
            all_logits = all_logits[new_idx]
            for j in range(batch_size):
                logit = all_logits[j*n_image:(j+1)*n_image]
                energy_var = (-np.log(np.exp(logit.data.cpu()).sum(1))).var()
                energy_Var.append(energy_var)
                energyVar = energy_var / (origin_Energy[idx[j]]*origin_Energy[idx[j]])     
                energyVar_values.append(energyVar.numpy())
                preds = torch.nn.functional.softmax(logit.detach().cpu(), dim=1)
                prob = preds.mean(0)
                y_pred = preds.sum(0)  
                y_pred = int(y_pred.max(0)[1].cpu()) 
                var = preds.var(0).sum()
                entropy = (-prob*np.log2(prob)).sum()
                entropy_values.append(entropy)
                tta_values.append(var)

        return np.array(origin_Energy).sum(), np.array(energyVar_values).sum(), np.array(tta_values).sum(), np.array(entropy_values).sum(),np.array(energy_Var).sum()

    def save_tta_values(self,tta,loss, iteration,train,predict):
        file_name = '_TTA_%s.txt'%self.args.dataset
        f = open(os.path.join(self.args.save_path, self.args.strategy+file_name),'a')
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(int(100*labeled/len(self.X)))
        f.write(str(labeled_percentage)+ '   ' + str(iteration) + '   ' + str(tta[0]) + '   ' + str(tta[1]) + '   ' + str(tta[2]) + '   ' + str(tta[3]) + '   ' + str(tta[4]) + '   '+ str(loss)  + '   ' + str(train) + '   '+ str(predict) + '\n')
        print('write in ',os.path.join(self.args.save_path, self.args.strategy+file_name))
        f.close()

    def coreset_coverage(self, embedding):
        """
            X: The whole dataset
            ib_idxes (Boolean array): The indexes of the selected labeled data
            output: max/average radius which cover the 100%/98% of the dataset
        """
        print('Coverage')
        lb_idxes = np.arange(self.n_pool)[self.idxs_lb]
        
        dist_ctr = pairwise_distances(embedding, embedding[lb_idxes])
        # group unlabeled data to their nearest labeled data
        min_args = np.argmin(dist_ctr, axis=1)
        print("min args: {}".format(min_args))
        delta = []
        for j in np.arange(len(lb_idxes)):
            # get the sample index for the jth center
            idxes = np.nonzero(min_args == j)[0]
            distances = dist_ctr[idxes, j]
            delta_j = 0 if len(distances)==0 else distances.max()
            delta.append(delta_j)
        # full cover
        coverage_mean = np.array(delta).mean()
        coverage_max = np.array(delta).max()
        coverage_topmean = np.sort(delta)[::-1][:int(len(delta)*0.3)].mean()
        return coverage_max, coverage_mean, coverage_topmean
    
    def collect_density(self,embedding,num,labelonly=True):
        print('Density')
        lb_idxes = np.arange(self.n_pool)[self.idxs_lb]
        if labelonly:
            dist_ctr = pairwise_distances(embedding[self.idxs_lb], embedding[self.idxs_lb])
        else:
            dist_ctr = pairwise_distances(embedding[self.idxs_lb], embedding)
        density = []
        for n in num:
            d = 0
            for j in range(len(lb_idxes)):
                distances = dist_ctr[j]
                distances.sort()
                d += distances[:n].mean()
            density.append(d)
        return density
    
    def save_coverage_density(self,coverage_max, coverage_mean, coverage_topmean,density,predict):
        file_name = '_Coverage_density_%s.txt'%self.args.dataset
        f = open(os.path.join(self.args.save_path, self.args.strategy+file_name),'a')
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(int(100*labeled/len(self.X)))
        f.write(str(labeled_percentage)+ '   ' + str('Load') + '   ' + str(coverage_max) + '   ' + str(coverage_mean) + '   ' + str(coverage_topmean)  + '   ' + str(density).replace(',','   ')[1:-1] + '   ' + str(predict) + '\n')
        print('write in ',os.path.join(self.args.save_path, self.args.strategy + file_name))
        f.close()

    def get_data_feature(self):
        from feature_model.get_feature import get_moco_feature
        loader_te = DataLoader(self.handler(self.X,self.Y, 
                        transform=self.args.transform_te), shuffle=False, pin_memory=True, **self.args.loader_te_args)
        if self.args.dataset == 'cifar10':
            moco_ckpt_path = '/research/dept2/yuli/dnn-testing/myTesting/DataCoverage/checkpoint/cifar10/ckpt_pretrained_mocov3/checkpoint.pth.tar'
        elif self.args.dataset == 'gtsrb':
            moco_ckpt_path = '/research/dept2/yuli/dnn-testing/myTesting/DataCoverage/checkpoint/gtsrb/ckpt_pretrained_mocov3/checkpoint.pth.tar'
        else:
            pass
        embedding,_ = get_moco_feature(loader_te, moco_ckpt_path)
        return embedding

    def save_all_coverage_density(self):
        import torchvision.models as models
        embedding = self.get_data_feature()
        
        import os
        files_names = []
        save_path = os.path.join(self.args.save_path,self.args.dataset)
        for root, dirs, files in os.walk(save_path, topdown=False):
            for name in files:
                if '.pkl' in name:
                    files_names.append(os.path.join(root, name).split('_')[-2])
        
        for i in files_names:
            self.args.load_model = str(i)
            if self.args.dataset == 'cifar10':
                self.far_load()
            else:
                self.load_model(is_small_budget=is_small_budget)
            test_acc = self.predict(self.X_te, self.Y_te)
            print('Load ', self.args.load_model)
            # embedding_lb = embedding[self.idxs_lb]
            density = self.collect_density(embedding,[2,5,10,25,50,100,200],labelonly=False)
            coverage_max, coverage_mean, coverage_topmean = self.coreset_coverage(embedding)
            print(coverage_max, coverage_mean, coverage_topmean,density,test_acc)
            self.save_coverage_density(coverage_max, coverage_mean, coverage_topmean,density,test_acc)


    def get_matrixs(self, dist_ctr):
        cl_idxs = np.load('embedding/%s_clusters.npy'%self.args.dataset)

        lb_idxes = np.arange(self.n_pool)[self.idxs_lb]
        # ul_idxes = np.arange(self.n_pool)[~self.idxs_lb]
        print('Compute Coverage Density! len', len(lb_idxes))
        coverage_dist_ctr = dist_ctr[:,lb_idxes]
        # coverage_dist_ctr_ul = dist_ctr[:,lb_idxes]
        # coverage_dist_ctr_ul = coverage_dist_ctr_ul[ul_idxes,:]

        min_args = np.argmin(coverage_dist_ctr, axis=1)
        delta = []
        for j in range(len(lb_idxes)):
            # get the sample index for the jth center
            idxes = np.nonzero(min_args == j)[0]
            distances = coverage_dist_ctr[idxes, j]
            delta_j = 0 if len(distances)==0 else distances.max()
            delta.append(delta_j)
        # full cover
        coverage_mean = np.array(delta).mean()
        coverage_max = np.array(delta).max()
        coverage_topmean = np.sort(delta)[::-1][:int(len(delta)*0.3)].mean()
        

        # cov_j = number of selected samples / total samples in the cluster
        # the standard deviation between cov_j
        all_data_cluster = [0 for i in range(1000)]
        selected_data_cluster = [0 for i in range(1000)]
        print ('cl_idxs length: ', len(cl_idxs))
        # plot the 
        for i,cl in enumerate(cl_idxs):
            if i in lb_idxes:
                selected_data_cluster[cl] += 1 
            all_data_cluster[cl] += 1 
        coverage_revised = (np.array(selected_data_cluster)/np.array(all_data_cluster)).var(0)
        
        density = []
        for n in [2,5,10,25,50,100,200]:
            d = 0
            for j in lb_idxes:
                distances = dist_ctr[j]
                distances.sort()
                d += distances[:n].mean()
            density.append(d)

        return density, coverage_max, coverage_mean, coverage_topmean,coverage_revised

    def save_metrixs(self, is_small_budget):
        embedding = np.load('embedding/%s_embed.npy'%self.args.dataset)
        if self.args.dataset == 'tinyimagenet':
            files_names = ['10.0','15.0','20.0','25.0','30.0','35.0','40.0','45.0','50.0','55.0','60.0','65.0','70.0']
        elif self.args.dataset == 'cifar10':
            files_names = [i*2+12 for i in range(24)]
        elif self.args.dataset == 'gtsrb':
            files_names = [i*2+12 for i in range(19)]
        else:
            files_names = [i+3 for i in range(18)]

        # model trained with samll budget
        if is_small_budget:
            files_names = list(np.arange(0.5, 4, 0.5))

        dist_ctr = pairwise_distances(embedding, embedding)
        for name in files_names:
            self.args.load_model = str(name)
            self.load_model(is_small_budget=is_small_budget) 
            test_acc = self.predict(self.X_te, self.Y_te)
            print('Load ', self.args.load_model)
            density, coverage_max, coverage_mean, coverage_topmean, coverage_cluster = self.get_matrixs(dist_ctr)
            self.save_metrixs_tofiles(density + [coverage_max, coverage_mean, coverage_topmean, coverage_cluster, test_acc])

    def save_metrixs_tofiles(self, metrixs):
        file_name = 'Metrixs_%s.csv'%self.args.dataset
        f = open(os.path.join(self.args.save_path, file_name),'a')
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(round(100*labeled/len(self.X), 1))
        metrixs = [self.args.strategy] + [labeled_percentage,] + metrixs + ['\n',]
        f.write((' ').join([str(item) for item in metrixs]))
        print('write in ', os.path.join(self.args.save_path, file_name))
        f.close()

