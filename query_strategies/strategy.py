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
from utils import print_log, time_string, AverageMeter, RecorderMeter, convert_secs2time
import time

torch.backends.cudnn.benchmark = True

class Strategy:
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        self.X = X  # vector
        self.Y = Y
        self.idxs_lb = idxs_lb # bool type
        self.handler = handler
        self.args = args
        self.n_pool = len(Y)
        self.pretrained = args.pretrained
        if args.pretrained:
            self.preprocessing = args.preprocessing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clf = net.to(self.device)
        self.net = net.to(self.device)

        if self.pretrained: # use the latent vector of the inputs as training data
            self.X_p = self.get_pretrained_embedding(X, Y)
        

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

            # exit()
            optimizer.zero_grad()

            out, e1 = self.clf(x) if not self.pretrained else self.clf.module.classifier(x)
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
        
        self.clf =  self.net.apply(weight_reset)
        self.clf = nn.DataParallel(self.clf).to(self.device)
        parameters = self.clf.parameters() if not self.pretrained else self.clf.module.classifier.parameters()
        optimizer = optim.SGD(parameters, lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        

        epoch_time = AverageMeter()
        recorder = RecorderMeter(n_epoch)
        epoch = 0 
        train_acc = 0.
        previous_loss = 0.
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr if not self.pretrained else None

            loader_tr = DataLoader(self.handler(self.X[idxs_train] if not self.pretrained else self.X_p[idxs_train], 
                                    torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
                                    transform=transform), shuffle=True, 
                                    **self.args.loader_tr_args)

            for epoch in range(n_epoch):
                ts = time.time()
                current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
                
                # Display simulation time
                need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
                need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
                
                # train one epoch
                train_acc, train_los = self._train(epoch, loader_tr, optimizer)

                # measure elapsed time
                epoch_time.update(time.time() - ts)

                print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                                   need_time, current_learning_rate
                                                                                   ) \
                + ' [Best : Train Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(True),
                                                               1. - recorder.max_accuracy(True)), self.args.log)
                
                
                recorder.update(epoch, train_los, train_acc, 0, 0)

                # The converge condition 
                if abs(previous_loss - train_los) < 0.001:
                    break
                else:
                    previous_loss = train_los

            self.clf = self.clf.module
        best_train_acc = recorder.max_accuracy(istrain=True)
        return best_train_acc                


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

    def predict_prob(self, X, Y):
        transform = self.args.transform_te if not self.pretrained else self.preprocessing
        loader_te = DataLoader(self.handler(X, Y, 
                        transform=transform), shuffle=False, **self.args.loader_te_args)

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
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
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
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
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
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
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
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
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
        loader_te = DataLoader(self.handler(X, Y, transform=transform),
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


def adjust_learning_rate(optimizer, epoch, gammas, schedule, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    "Add by YU"
    lr = args.lr
    mu = args.momentum

    if args.optimizer != "YF":
        assert len(gammas) == len(
            schedule), "length of gammas and schedule should be equal"
        for (gamma, step) in zip(gammas, schedule):
            if (epoch >= step):
                lr = lr * gamma
            else:
                break
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    elif args.optimizer == "YF":
        lr = optimizer._lr
        mu = optimizer._mu

    return lr, mu

