from joblib.externals.cloudpickle.cloudpickle import instance
import numpy as np
from sklearn import preprocessing
from torch import nn
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from copy import deepcopy
from utils import print_log, time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import time
import sys, os
from torchvision.utils import save_image
from tqdm import tqdm
from .util import AugMixDataset,TransformTwice,linear_rampup,interleave_offsets,interleave,WeightEMA

torch.backends.cudnn.benchmark = True

lambda_u = 100
ema_decay = 0.999
T = 0.5
alpha = 0.75

class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        # print("in loss",outputs_x.size(0),targets_x.size(0))
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)

        return Lx, Lu, lambda_u * linear_rampup(epoch)

class semi_Strategy:
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
        self.clf = net[0].to(self.device)
        self.net = net[0].to(self.device)

        if self.pretrained: # use the latent vector of the inputs as training data
            self.X_p = self.get_pretrained_embedding(X, Y)

        self.ema_model = net[1].to(self.device)
        for param in self.ema_model.parameters():
            param.detach_()
        
    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = idxs_lb

    def _train(self, epoch, loader_labeled, loader_unlabeled, optimizer, ema_optimizer, train_iteration):
        self.clf.train()
        accFinal = 0.
        train_loss = 0.
        labeled_train_iter = iter(loader_labeled)
        unlabeled_train_iter = iter(loader_unlabeled)
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()
        criterion = SemiLoss()
        ce_loss = 0.
        acc_divisor = 1e-5
        for batch_idx in range(int(train_iteration)):
           
            try:
                inputs_x, targets_x, _ = labeled_train_iter.next()
                out, e1 = self.clf(inputs_x.to(self.device))
                loss = F.cross_entropy(out, targets_x.to(self.device))

                ce_loss += loss.item()
                accFinal += torch.sum((torch.max(out,1)[1] == targets_x.to(self.device)).float()).data.item()
                batch_size = inputs_x.size(0)
                targets_x = torch.zeros(batch_size, self.args.n_class).scatter_(1, targets_x.view(-1,1).long(), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device) 
            except:
                labeled_train_iter = iter(loader_labeled)
                inputs_x, targets_x, _ = labeled_train_iter.next()
                out, e1 = self.clf(inputs_x.to(self.device))
                loss = F.cross_entropy(out, targets_x.to(self.device))
                Store_TTA = False
                ce_loss += loss.item()
                accFinal += torch.sum((torch.max(out,1)[1] == targets_x.to(self.device)).float()).data.item()
                batch_size = inputs_x.size(0)
                targets_x = torch.zeros(batch_size, self.args.n_class).scatter_(1, targets_x.view(-1,1).long(), 1)
                inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)
            
            

            try:
                (inputs_u,inputs_u2), _ , _ = unlabeled_train_iter.next()
                inputs_u = inputs_u.to(self.device) 
                inputs_u2 = inputs_u2.to(self.device)
            except:
                unlabeled_train_iter = iter(loader_unlabeled)
                (inputs_u,inputs_u2), _ , _ = unlabeled_train_iter.next()
                inputs_u = inputs_u.to(self.device) 
                inputs_u2 = inputs_u2.to(self.device)

            with torch.no_grad():
                # compute guessed labels of unlabel samples
                outputs_u,_ = self.clf(inputs_u)
                outputs_u2,_ = self.clf(inputs_u2)
                p = (torch.softmax(outputs_u, dim=1) + torch.softmax(outputs_u2, dim=1)) / 2
                pt = p**(1/T)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            acc_divisor += batch_size
            if batch_size != inputs_u.size(0):
                continue
            # print(batch_size,inputs_u.size(0))
            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u, inputs_u2], dim=0)
            all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

            l = np.random.beta(alpha, alpha)

            l = max(l, 1-l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b
            # print(mixed_input.size(0))
            # interleave labeled and unlabed samples between batches to get correct batchnorm calculation 
            mixed_input = list(torch.split(mixed_input, batch_size))
            # print(len(mixed_input),len(mixed_input[0]),len(mixed_input[1]))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [self.clf(mixed_input[0])[0]]

            for input in mixed_input[1:]:
                logits.append(self.clf(input)[0])

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)

            Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:], epoch+batch_idx/train_iteration)

            loss = Lx + w * Lu
            train_loss += loss.item()
            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))
            ws.update(w, inputs_x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            
            
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)

            if batch_idx % 10 == 0:
                print ("[Batch={:03d}] [Loss={:.2f}]".format(batch_idx, loss))

        return accFinal / acc_divisor, train_loss, ce_loss/int(train_iteration)

    
    def train(self, alpha=0.1, n_epoch=10):

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear): 
                m.reset_parameters()
        
        self.clf =  self.net.apply(weight_reset)
        self.ema_model =  self.ema_model.apply(weight_reset)
        
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters() if not self.pretrained else self.clf.module.classifier.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        ema_optimizer= WeightEMA(self.clf, self.ema_model, alpha=ema_decay, lr = self.args.lr)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        previous_loss = 0.
        train_iteration = len(self.X) / self.args.loader_tr_args['batch_size']
        
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr if not self.pretrained else None

            loader_labeled = DataLoader(self.handler(self.X[idxs_train] if not self.pretrained else self.X_p[idxs_train], 
                                    torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform), shuffle=True, 
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    **self.args.loader_tr_args)
            # print(len(loader_labeled.dataset.X))
            # exit()
            loader_unlabeled = DataLoader(self.handler(self.X[idxs_unlabeled] if not self.pretrained else self.X_p[idxs_unlabeled], 
                                    torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(), 
                                    transform=TransformTwice(transform)), shuffle=True, 
                                    **self.args.loader_tr_args)
            # train_iteration = max(len(loader_labeled.dataset.X),len(loader_unlabeled.dataset.X))/self.args.loader_tr_args['batch_size']
            
            # print('X',len(self.X))
            # print('has:', n_epoch)
            for epoch in range(n_epoch):
                print(epoch,n_epoch)
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                ema_optimizer.set_wd(current_learning_rate)

                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los, ce_loss = self._train(epoch, loader_labeled, loader_unlabeled, optimizer, ema_optimizer,train_iteration)
                
                # measure elapsed time
                epoch_time.update(time.time() - ts)

                print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Train Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(True),
                                                               1. - recorder.max_accuracy(True)), self.args.log)
                
                
                test_acc = self.predict(self.X_te, self.Y_te)
                
                recorder.update(epoch, train_los, train_acc, 0, test_acc)
                # The converge condition 
                if abs(previous_loss - train_los) < 0.0001:
                    break
                else:
                    previous_loss = train_los

            self.clf = self.clf.module
            if self.args.save_model:
                self.save_model()

        if self.args.save_tta:
            test_acc = self.predict(self.X_te, self.Y_te)
            self.save_tta_values(self.get_tta_values(),train_los,epoch, train_acc, test_acc)

        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc                


    def predict(self, X, Y):
        # add support for pretrained model
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        
        if not self.pretrained:
            self.ema_model.eval()
        else:
            self.ema_model.classifier.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.ema_model(x)
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
            self.ema_model.eval()
        else:
            self.ema_model.classifier.eval()

        correct = 0
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.ema_model(x)
                pred = out.max(1)[1]     
                P[idxs] = pred           
                correct +=  (y == pred).sum().item() 
   
        return P

    def predict_prob(self, X, Y):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, **self.args.loader_te_args)

        if not self.pretrained:
            self.ema_model.eval()
        else:
            self.ema_model.classifier.eval()

        probs = torch.zeros([len(Y), len(np.unique(self.Y))])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                out, e1 = self.ema_model(x)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu().data
        
        return probs

    def predict_prob_dropout(self, X, Y, n_drop):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        if not self.pretrained:
            self.ema_model.train()
        else:
            self.ema_model.classifier.train()

        probs = torch.zeros([len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.ema_model(x)
                    prob = F.softmax(out, dim=1)
                    probs[idxs] += prob.cpu().data
        probs /= n_drop
        
        return probs

    def predict_prob_dropout_split(self, X, Y, n_drop):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)

        if not self.pretrained:
            self.ema_model.train()
        else:
            self.ema_model.classifier.train()

        probs = torch.zeros([n_drop, len(Y), len(np.unique(Y))])
        with torch.no_grad():
            for i in range(n_drop):
                print('n_drop {}/{}'.format(i+1, n_drop))
                for x, y, idxs in loader_te:
                    x, y = x.to(self.device), y.to(self.device) 
                    out, e1 = self.ema_model(x)
                    probs[i][idxs] += F.softmax(out, dim=1).cpu().data
            return probs

    def seed_worker(self, worker_id):
        """
        To preserve reproducibility when num_workers > 1
        """
        import random
        # https://pytorch.org/docs/stable/notes/randomness.html
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_embedding(self, X, Y):
        """ get last layer embedding from current model"""
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        if not self.pretrained:
            self.ema_model.eval()
        else:
            self.ema_model.classifier.eval()
        
        embedding = torch.zeros([len(Y), 
                self.ema_model.module.get_embedding_dim() if isinstance(self.ema_model, nn.DataParallel) 
                else self.ema_model.get_embedding_dim()])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.ema_model(x)
                embedding[idxs] = e1.data.cpu().float()
        
        return embedding

    def get_pretrained_embedding(self, X, Y):
        """ get embedding from the pretrained model: add by Yuli
            Only valid if pretrained is true
        """
        if not self.pretrained:
            raise ValueError("pretrained is not true")

        transform = self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
    
        embedding = torch.zeros([len(Y), 512])
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                e1 = self.ema_model.fe.encode_image(x)
                embedding[idxs] = e1.data.cpu().float()
        
        return embedding
    
    def get_grad_embedding(self, X, Y):
        """ gradient embedding (assumes cross-entropy loss) of the last layer"""
        transform = self.args.transform_te if not self.pretrained else self.preprocessing

        model = self.ema_model
        if isinstance(model, nn.DataParallel):
            model = model.module
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(np.unique(Y))
        embedding = np.zeros([len(Y), embDim * nLab])
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
                            shuffle=False, **self.args.loader_te_args)
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                cout, out = self.ema_model(x)
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
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(float(100*labeled/len(self.X)))
        torch.save(self.ema_model, os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.manualSeed)+'.pkl'))
        path = os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_'+str(self.args.manualSeed)+'.npy')
        np.save(path,self.idxs_lb)
        print('save to ',os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage))

    def load_model(self):
        self.ema_model = torch.load(os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+str(self.args.load_model)+'.pkl'))

    def get_tta_values(self,save_info=False):    
        # Save Energy, Energy Variance, Smoothness, Entropy for analysis 
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(self.X_te, self.Y_te, transform=transform), pin_memory=True, 
                        shuffle=False, **self.args.loader_te_args)
        self.ema_model.eval()
        origin_Energy = torch.zeros(len(self.Y_te))
        origin_Entropy = torch.zeros(len(self.Y_te))
        origin_Confidence = torch.zeros(len(self.Y_te))
        origin_Margin = torch.zeros(len(self.Y_te))
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device) 
                out, e1 = self.ema_model(x)
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
        self.ema_model.eval()
        new_idx = []
        for k in range(batch_size):
            for j in range(n_image):
                new_idx.append(k+j*batch_size)
        for i, (all_images,idx) in tqdm(enumerate(augtest_loader)):
            all_logits, _ = self.ema_model(torch.cat(all_images, 0).to(self.device))
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
        f = open(os.path.join(self.args.save_path, self.args.strategy+'_TTA.txt'),'a')
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(int(100*labeled/len(self.X)))
        f.write(str(labeled_percentage)+ '   ' + str(iteration) + '   ' + str(tta[0]) + '   ' + str(tta[1]) + '   ' + str(tta[2]) + '   ' + str(tta[3]) + '   ' + str(tta[4]) + '   '+ str(loss)  + '   ' + str(train) + '   '+ str(predict) + '\n')
        print('write in ',os.path.join(self.args.save_path, self.args.strategy+'_TTA.txt'))
        f.close()



