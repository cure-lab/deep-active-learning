import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torch.autograd import grad
from .strategy import Strategy
from utils import time_string, AverageMeter, RecorderMeter, convert_secs2time, adjust_learning_rate
import os
# The original implenmentation is from: https://github.com/cjshui/WAAL/blob/7104bc0113e7e73218a6d287951642a76d8005df/query_strategies/wasserstein_adversarial.py#L40


# setting gradient values
def set_requires_grad(model, requires_grad=True):
    """
    Used in training adversarial approach
    :param model:
    :param requires_grad:
    :return:
    """

    for param in model.parameters():
        param.requires_grad = requires_grad


# setting gradient penalty for sure the lipschitiz property
def gradient_penalty(critic, h_s, h_t, device):
    ''' Gradeitnt penalty approach'''
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.cat([interpolates, h_s, h_t]).requires_grad_()
    # interpolates.requires_grad_()
    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()

    return gradient_penalty


class WAAL(Strategy):
    def __init__(self,X,Y, X_te, Y_te, idxs_lb, net, 
                train_handler, test_handler, args):
        super(WAAL, self).__init__(X, Y,  X_te, Y_te, idxs_lb, net, test_handler, args)
        """
        :param X:
        :param Y:
        :param idxs_lb:
        :param net_fea:
        :param net_clf:
        :param net_dis:
        :param train_handler: generate a dataset in the training procedure, since training requires two datasets, the returning value
                                looks like a (index, x_dis1, y_dis1, x_dis2, y_dis2)
        :param test_handler: generate a dataset for the prediction, only requires one dataset
        :param args:
        """

        self.X = X
        self.Y = Y
        self.X_te = X_te
        self.Y_te = Y_te
        self.idxs_lb  = idxs_lb
        self.net_fea = net.feature_extractor
        self.net_clf = net.linear
        self.net_dis = net.discriminator
        self.train_handler = train_handler
        self.test_handler  = test_handler
        self.args    = args

        self.n_pool  = len(Y)
        self.num_class = self.args.n_class
        use_cuda     = torch.cuda.is_available()
        self.device  = torch.device("cuda:0" if use_cuda else "cpu")

        self.selection = 10
        # for cifar 10 or svhn or fashion mnist  self.selection = 10


    def train(self, alpha, n_epoch):

        """
        Only training samples with labeled and unlabeled data-set
        alpha is the trade-off between the empirical loss and error, the more interaction, the smaller \alpha
        :return:
        """

        print("[Training] labeled and unlabeled data")

        self.fea = self.net_fea.to(self.device)
        self.clf = self.net_clf.to(self.device)
        self.dis = self.net_dis.to(self.device)


        # setting three optimizers

        opt_fea = optim.SGD(self.fea.parameters(),lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        opt_clf = optim.SGD(self.clf.parameters(),lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        opt_dis = optim.SGD(self.dis.parameters(),lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)

        # setting idxs_lb and idx_ulb
        idxs_lb_train = np.arange(self.n_pool)[self.idxs_lb]
        idx_ulb_train = np.arange(self.n_pool)[~self.idxs_lb]

        # computing the unbalancing ratio, a value betwwen [0,1], generally 0.1 - 0.5
        gamma_ratio = len(idxs_lb_train)/len(idx_ulb_train)
        # gamma_ratio = 1

        # Data-loading (Redundant Trick)
        transform = self.args.transform_tr        
        loader_tr = DataLoader(self.train_handler(self.X[idxs_lb_train], 
                self.Y[idxs_lb_train], self.X[idx_ulb_train], self.Y[idx_ulb_train],
                transform = transform), 
                shuffle= True, **self.args.loader_tr_args)

        recorder = RecorderMeter(n_epoch)
        epoch_time = AverageMeter()
        accCurrent = 0.
        accOld = 0.
        
        for epoch in range(n_epoch):
            ts = time.time()
            current_lr_fea, _ = adjust_learning_rate(opt_fea, epoch, self.args.gammas, self.args.schedule, self.args)
            current_lr_clf, _ = adjust_learning_rate(opt_clf, epoch, self.args.gammas, self.args.schedule, self.args)
            current_lr_dis, _ = adjust_learning_rate(opt_dis, epoch, self.args.gammas, self.args.schedule, self.args)
            
            # Display simulation time
            need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (n_epoch - epoch))
            need_time = '[{} Need: {:02d}:{:02d}:{:02d}]'.format(self.args.strategy, need_hour, need_mins, need_secs)
                
            # print('==========Inner epoch {:d} ========'.format(epoch))

            # setting the training mode in the beginning of EACH epoch
            # (since we need to compute the training accuracy during the epoch, optional)

            self.fea.train()
            self.clf.train()
            self.dis.train()


            Total_loss = 0
            n_batch    = 0
            acc        = 0

            for index, label_x, label_y, unlabel_x, _ in loader_tr:
                n_batch += 1

                label_x, label_y = label_x.to(self.device), label_y.to(self.device)
                unlabel_x = unlabel_x.to(self.device)


                # training feature extractor and predictor

                set_requires_grad(self.fea,requires_grad=True)
                set_requires_grad(self.clf,requires_grad=True)
                set_requires_grad(self.dis,requires_grad=False)

                lb_z, _   = self.fea(label_x)
                unlb_z, _ = self.fea(unlabel_x)

                opt_fea.zero_grad()
                opt_clf.zero_grad()
                lb_out, _ = self.clf(lb_z)
                # prediction loss (deafult we use F.cross_entropy)
                pred_loss = torch.mean(F.cross_entropy(lb_out,label_y))


                # Wasserstein loss (here is the unbalanced loss, because we used the redundant trick)
                wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()


                with torch.no_grad():

                    lb_z, _ = self.fea(label_x)
                    unlb_z, _ = self.fea(unlabel_x)

                gp = gradient_penalty(self.dis, unlb_z, lb_z, device=self.device)

                loss = pred_loss + alpha * wassertein_distance + alpha * gp * 5
                # for CIFAR10 the gradient penality is 5
                # for SVHN the gradient penality is 2
                # 

                loss.backward()
                # print ('--- batch {}, loss {} ---'.format(n_batch, loss.item()))
                # print (pred_loss.item(), wassertein_distance.item(), gp.item())
                opt_fea.step()
                opt_clf.step()


                # Then the second step, training discriminator

                set_requires_grad(self.fea, requires_grad=False)
                set_requires_grad(self.clf, requires_grad=False)
                set_requires_grad(self.dis, requires_grad=True)


                with torch.no_grad():

                    lb_z, _ = self.fea(label_x)
                    unlb_z, _ = self.fea(unlabel_x)


                for _ in range(1):

                    # gradient ascent for multiple times like GANS training

                    gp = gradient_penalty(self.dis, unlb_z, lb_z, device=self.device)

                    wassertein_distance = self.dis(unlb_z).mean() - gamma_ratio * self.dis(lb_z).mean()

                    dis_loss = -1 * alpha * wassertein_distance - alpha * gp * 2

                    opt_dis.zero_grad()
                    dis_loss.backward()
                    opt_dis.step()


                # prediction and computing training accuracy and empirical loss under evaluation mode
                P = lb_out.max(1)[1]
                acc += 1.0 * (label_y == P).sum().item() / len(label_y)
                Total_loss += loss.item()

            Total_loss /= n_batch
            acc        /= n_batch
            
            # test acc
            test_acc = self.predict(self.X_te, self.Y_te)
            
            epoch_time.update(time.time() - ts)
            recorder.update(epoch, Total_loss, acc, 0, test_acc)
            # print('Training Loss {:.3f}'.format(Total_loss))
            # print('Training accuracy {:.3f}'.format(acc*100))
            print('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [LR={:6.4f}]'.format(time_string(), epoch, n_epoch,
                                                                        need_time, current_lr_fea
                                                                        ) \
                + ' [Best : Test Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                    1. - recorder.max_accuracy(False)))
    
    
            accCurrent = acc
            # if abs(accCurrent - accOld) <= 0.0005:
            #     break
            # else:
            #     accOld = accCurrent

        recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
        test_acc = recorder.max_accuracy(istrain=False)
        return test_acc
        
    def predict(self,X,Y):
        print ("start to predict...")
        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args.transform_te),
                               shuffle=False, **self.args.loader_te_args)

        self.fea.eval()
        self.clf.eval()

        # P = torch.zeros(len(Y), dtype=Y.dtype)
        correct = 0.
        with torch.no_grad():
            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent,_  = self.fea(x)
                out, _  = self.clf(latent)
                pred    = out.max(1)[1]
                # P[idxs] = pred.cpu()
                correct +=  (y == pred).sum().item() 
                
            test_acc = 1. * correct / len(Y)

        return test_acc


    def predict_prob(self,X,Y):

        """
        prediction output score probability
        :param X:
        :param Y: NEVER USE the Y information for direct prediction
        :return:
        """

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args.transform_te),
                               shuffle=False, **self.args.loader_te_args)

        self.fea.eval()
        self.clf.eval()

        probs = torch.zeros([len(Y), self.num_class])
        with torch.no_grad():

            for x, y, idxs in loader_te:
                x, y = x.to(self.device), y.to(self.device)
                latent, _ = self.fea(x)
                out, _ = self.clf(latent)
                prob = F.softmax(out, dim=1)
                probs[idxs] = prob.cpu()

        return probs


    def pred_dis_score(self,X,Y):

        """
        prediction discrimnator score
        :param X:
        :param Y:  FOR numerical simplification, NEVER USE Y information for prediction
        :return:
        """

        loader_te = DataLoader(self.test_handler(X, Y, transform=self.args.transform_te),
                               shuffle=False, **self.args.loader_te_args)

        self.fea.eval()
        self.dis.eval()

        scores = torch.zeros(len(Y))

        with torch.no_grad():
            for x, y, idxs in loader_te:

                x, y = x.to(self.device), y.to(self.device)
                latent,_ = self.fea(x)
                out = self.dis(latent).cpu()
                scores[idxs] = out.view(-1)

        return scores


    def single_worst(self, probas):

        """
        The single worst will return the max_{k} -log(proba[k]) for each sample
        :param probas:
        :return:  # unlabeled \times 1 (tensor float)
        """

        value,_ = torch.max(-1*torch.log(probas),1)

        return value


    def L2_upper(self, probas):

        """
        Return the /|-log(proba)/|_2
        :param probas:
        :return:  # unlabeled \times 1 (float tensor)
        """

        value = torch.norm(torch.log(probas),dim=1)

        return value


    def L1_upper(self, probas):

        """
        Return the /|-log(proba)/|_1
        :param probas:
        :return:  # unlabeled \times 1
        """
        value = torch.sum(-1*torch.log(probas),dim=1)

        return value


    def query(self,query_num):

        """
        adversarial query strategy
        :param n:
        :return:
        """

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

        # prediction output probability
        probs = self.predict_prob(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])



        # uncertainly score (three options, single_worst, L2_upper, L1_upper)
        # uncertainly_score = self.single_worst(probs)
        uncertainly_score = 0.5* self.L2_upper(probs) + 0.5* self.L1_upper(probs)

        # print(uncertainly_score)


        # prediction output discriminative score
        dis_score = self.pred_dis_score(self.X[idxs_unlabeled], self.Y[idxs_unlabeled])

        # print(dis_score)


        # computing the decision score
        total_score = uncertainly_score - self.selection * dis_score
        # print(total_score)
        b = total_score.sort()[1][:query_num]
        # print(total_score[b])


        # sort the score with minimal query_number examples
        # expected value outputs from smaller to large

        return idxs_unlabeled[total_score.sort()[1][:query_num]]
