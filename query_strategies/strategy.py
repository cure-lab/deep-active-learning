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
        previous_loss = 0.
        if idxs_train.shape[0] != 0:
            transform = self.args.transform_tr if not self.pretrained else None

            train_data = self.handler(self.X[idxs_train] if not self.pretrained else self.X_p[idxs_train], 
                                torch.Tensor(self.Y.numpy()[idxs_train]).long(), 
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

                # The converge condition 
                if abs(previous_loss - train_los) < 0.0005:
                    break
                else:
                    previous_loss = train_los
                    
            if self.args.save_model:
                self.save_model()
            recorder.plot_curve(os.path.join(self.args.save_path, self.args.dataset))
            self.clf = self.clf.module
            # self.save_tta_values(self.get_tta_values())

        best_test_acc = recorder.max_accuracy(istrain=False)
        return best_test_acc                


    def predict(self, X, Y):
        # add support for pretrained model
        transform=self.args.transform_te if not self.pretrained else self.preprocessing
        if type(X) is np.ndarray:
            loader_te = DataLoader(self.handler(X, Y, transform=transform), pin_memory=True, 
                            shuffle=False, **self.args.loader_te_args)
        else: 
            loader_te = DataLoader(self.handler(X.numpy(), Y, transform=transform), pin_memory=True,
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


    # def get_tta_values(self):    
    #     # apply multiple data augmentation and calculate the variance
    #     # TO BE TESTED
    #     from torchvision.utils import make_grid

    #     n_class, n_iters = 10, 32 
    #     tta_values = []
    #     augset = AugMixDataset(self.X_te, self.args.transform_te, n_iters) 
    #     augtest_loader = torch.utils.data.DataLoader(
    #                     augset,
    #                     batch_size=1,
    #                     shuffle=False,
    #                     num_workers=self.args.loader_te_args['num_workers'],
    #                     pin_memory=True)
    #     self.clf.eval()
    #     for i, all_images in tqdm(enumerate(augtest_loader)):
    #         # print("progress {}/{}".format(i, len(self.X_te)))
    #         all_logits, _ = self.clf(torch.cat(all_images, 0).to(self.device))
    #         preds = torch.nn.functional.softmax(all_logits.detach().cpu(), dim=1)
    #         prob = preds.mean(0)
    #         entropy = (-prob*np.log2(prob)).sum()
    #         tta_values.append(entropy)
    #         # variance = preds.var(0).sum()
    #         # tta_values.append(variance)
    #         if i % 1000 == 0:
                
    #             # print ("size: ", len(all_images))
    #             # print ("tensor size: ", torch.cat(all_images, 0).shape)
    #             grid = make_grid(torch.cat(all_images, 0), nrow=len(all_images), padding=2)
    #             save_image(grid, fp=os.path.join(self.args.save_path, 'img_augmentation_{}.png'.format(i)))

    #     print('TTA_values:', np.array(tta_values).sum())
    #     return np.array(tta_values).sum()
    
    def save_tta_values(self,tta):
        f = open(os.path.join(self.args.save_path, self.args.strategy+'_tta_value.txt'),'a')
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        f.write(str(labeled)+ '   ' + str(tta)+ '\n')
        print('write in ',os.path.join(self.args.save_path, self.args.strategy+'_tta_value.txt'))
        f.close()
    
    def save_model(self):
        labeled = len(np.arange(self.n_pool)[self.idxs_lb])
        labeled_percentage = str(int(100*labeled/len(self.X)))
        torch.save(self.clf, os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'.pkl'))
        print('save to ',os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+labeled_percentage+'_parameter.pkl'))

    def load_model(self):
        self.clf = torch.load(os.path.join(self.args.save_path, self.args.strategy+'_'+self.args.model+'_'+str(self.args.load_model)+'.pkl'))

    def tta_test_from_load_model(self):
        if self.args.load_model != 0:
            self.load_model()
        test_acc = self.predict(self.X_te, self.Y_te)
        tta = self.get_tta_values()
        print(test_acc, tta)
        
class AugMixDataset(torch.utils.data.Dataset):
    """Dataset wrapper to perform AugMix augmentation."""
    def __init__(self, dataset, preprocess, n_iters, no_jsd=False):
        self.dataset = dataset
        self.preprocess = preprocess
        self.no_jsd = no_jsd
        self.n_iters = n_iters

    def __getitem__(self, i):
        x = self.dataset[i]
        if self.no_jsd:
            return aug(x, self.preprocess)
        else:
            aug_image = [aug(x, self.preprocess) for i in range(self.n_iters)]
            im_tuple = [self.preprocess(x)] + aug_image
            return im_tuple

    def __len__(self):
        return len(self.dataset)

def aug(image, preprocess):
    """Perform AugMix augmentations and compute mixture.
    Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
    Returns:
        mixed: Augmented and mixed image.
    """
    from . import augmentations
    from PIL import Image
    image = Image.fromarray(image)
    aug_list = augmentations.augmentations
    # if args.all_ops:
    #     aug_list = augmentations.augmentations_all
    mixture_width, mixture_depth = 1, -1
    aug_severity = 3
    ws = np.float32(np.random.dirichlet([1] * mixture_width))
    m = np.float32(np.random.beta(1, 1))

    mix = torch.zeros_like(preprocess(image))
    for i in range(mixture_width):
        image_aug = image.copy()
        depth = mixture_depth if mixture_depth > 0 else np.random.randint(
                1, 4)
        for _ in range(depth):
            op = np.random.choice(aug_list)
            image_aug = op(image_aug, aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image_aug)

    mixed = (1 - m) * preprocess(image) + m * mix
    return np.array(mixed)
