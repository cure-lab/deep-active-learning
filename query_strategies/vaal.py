'''
@article{sinha2019variational,
  title={Variational Adversarial Active Learning},
  author={Sinha, Samarth and Ebrahimi, Sayna and Darrell, Trevor},
  journal={arXiv preprint arXiv:1904.00370},
  year={2019}
}
CodeBase: https://github.com/sinhasam/vaal
'''
import numpy as np
from torch.cuda import device
from .strategy import Strategy
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from sklearn.metrics import accuracy_score
import copy
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils import print_log, time_string, AverageMeter, RecorderMeter, convert_secs2time

class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAE_MNIST(nn.Module):
    def __init__(self, x_dim=784, h_dim1=512, h_dim2=256, z_dim=2):
        super(VAE_MNIST, self).__init__()
        
        # encoder part
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim2)
        self.fc31 = nn.Linear(h_dim2, z_dim)
        self.fc32 = nn.Linear(h_dim2, z_dim)
        # decoder part
        self.fc4 = nn.Linear(z_dim, h_dim2)
        self.fc5 = nn.Linear(h_dim2, h_dim1)
        self.fc6 = nn.Linear(h_dim1, x_dim)
        
    def encoder(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc31(h), self.fc32(h) # mu, log_var
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu) # return z sample
        
    def decoder(self, z):
        h = F.relu(self.fc4(z))
        h = F.relu(self.fc5(h))
        h = F.sigmoid(self.fc6(h)) 
        h = h.view(-1, 1, 28, 28)
        return h
    
    def forward(self, x):
        # print (x.shape)
        mu, log_var = self.encoder(x.view(-1, 784))
        z = self.sampling(mu, log_var)
        return self.decoder(z), z, mu, log_var


class Discriminator_MNIST(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=2):
        super(Discriminator_MNIST, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 5),
            nn.ReLU(True),
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                self.kaiming_init(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, z):
        return self.net(z)
        

class VAE(nn.Module):
    """Encoder-Decoder architecture for both WAE-MMD and WAE-GAN."""
    def __init__(self, z_dim=32, nc=3):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.in_dim = z_dim//32
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 128, 4, 2, 1, bias=False),  # B,  128, 32, 32
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # B,  512,  8,  8
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # B, 1024,  4,  4
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            View((-1, 1024 * 2 * self.in_dim * 2 * self.in_dim)),  # B, 1024*4*4
        )

        self.fc_mu = nn.Linear(1024 * 2 * self.in_dim * 2 * self.in_dim, z_dim)  # B, z_dim
        self.fc_logvar = nn.Linear(1024 * 2 * self.in_dim * 2 * self.in_dim, z_dim)  # B, z_dim
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 1024 * 4 * self.in_dim * 4 * self.in_dim),  # B, 1024*8*8
            View((-1, 1024, 4 * self.in_dim, 4 * self.in_dim)),  # B, 1024,  8,  8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),  # B,  512, 16, 16
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),  # B,  256, 32, 32
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # B,  128, 64, 64
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, nc, 1),  # B,   nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            try:
                for m in self._modules[block]:
                    self.kaiming_init(m)
            except:
                self.kaiming_init(block)

    def forward(self, x):
        # print('x',x.shape)
        z = self._encode(x)
        # print('z',z.shape)
        mu, logvar = self.fc_mu(z), self.fc_logvar(z)
        z = self.reparameterize(mu, logvar)
        # print('z', z.shape)
        x_recon = self._decode(z)
        # print('x_recon', x_recon.shape)

        return x_recon, z, mu, logvar

    def reparameterize(self, mu, logvar):
        stds = (0.5 * logvar).exp()
        epsilon = torch.randn(*mu.size())
        # if mu.is_cuda:
        #     stds, epsilon = stds.to(device), epsilon.to(device)
        stds, epsilon = stds.to(device), epsilon.to(device)
        latents = epsilon * stds + mu
        return latents

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Discriminator(nn.Module):
    """Adversary architecture(Discriminator) for WAE-GAN."""
    def __init__(self, z_dim=10):
        super(Discriminator, self).__init__()
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                self.kaiming_init(m)

    def kaiming_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.weight.data.fill_(1)
            if m.bias is not None:
                m.bias.data.fill_(0)

    def forward(self, z):
        return self.net(z)





def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


class AdversarySampler:
    '''
    Sample top-k unlabeled data based on the discriminator outputs
    '''
    def __init__(self, budget):
        self.budget = budget

    def sample(self, vae, discriminator, data):
        all_preds = []
        all_indices = []

        for images, _, indices in data:
            images = images.to(device)

            with torch.no_grad():
                _, _, mu, _ = vae(images)
                preds = discriminator(mu)

            preds = preds.cpu().data
            all_preds.extend(preds)
            all_indices.extend(indices)

        all_preds = torch.stack(all_preds)
        all_preds = all_preds.view(-1)
        # need to multiply by -1 to be able to use torch.topk
        all_preds *= -1

        # select the points which the discriminator things are the most likely to be unlabeled
        _, querry_indices = torch.topk(all_preds, int(self.budget))
        querry_pool_indices = np.asarray(all_indices)[querry_indices]

        return querry_pool_indices


class VAAL(Strategy):
    def __init__(self, X, Y, X_te, Y_te,  idxs_lb, net, handler, args):
        super(VAAL, self).__init__(X, Y,  X_te, Y_te, idxs_lb, net, handler, args)
        global device
        device = self.device
        if self.args.channels == 3:
            self.vae = VAE(self.args.img_size).to(device)
            self.discriminator = Discriminator(self.args.img_size).to(device)
        else:
            self.vae = VAE_MNIST().to(device)
            self.discriminator = Discriminator_MNIST().to(device)
        
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        self.cuda = self.device
        self.num_vae_steps = 2
        self.beta = 1
        self.adversary_param = 1
        self.num_adv_steps = 1


    def vae_loss(self, x, recon, mu, logvar, beta):
        MSE = self.mse_loss(recon, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD = KLD * beta
        return MSE + KLD

    def read_data(self, dataloader, labels=True):
        if labels:
            while True:
                for img, label, _ in dataloader:
                    yield img, label
        else:
            while True:
                for img, _, _ in dataloader:
                    yield img

    def vaal_train(self, epoch, loader_tr, optimizer, labeled_data, unlabeled_data, optim_vae, optim_discriminator):
        self.clf.train()
        accFinal = 0.
        self.vae.train()
        self.discriminator.train()

        for batch_idx, (x, y, idxs) in enumerate(loader_tr):
            # x, y = Variable(x.to(device)), Variable(y.to(device))
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out, e1 = self.clf(x)
            loss = F.cross_entropy(out, y)
            accFinal += torch.sum((torch.max(out, 1)[1] == y).float()).data.item()
            loss.backward()
            # clamp gradients, just in case
            for p in filter(lambda p: p.grad is not None, self.clf.parameters()): p.grad.data.clamp_(min=-.1, max=.1)
            optimizer.step()

            labeled_imgs, labels = next(labeled_data)
            # print(x.shape,labeled_imgs.shape)
            unlabeled_imgs = next(unlabeled_data)

            labeled_imgs = labeled_imgs.to(device)
            unlabeled_imgs = unlabeled_imgs.to(device)
            labels = labels.to(device)

            # # task_model step
            # preds = task_model(labeled_imgs)
            # task_loss = self.ce_loss(preds, labels)
            # optim_task_model.zero_grad()
            # task_loss.backward()
            # optim_task_model.step()

            # VAE step
            for count in range(self.num_vae_steps):
                # print(labeled_imgs.shape)
                recon, z, mu, logvar = self.vae(labeled_imgs)
                # print(recon.shape,labeled_imgs.shape,z.shape)
                unsup_loss = self.vae_loss(labeled_imgs, recon, mu, logvar, self.beta)
                unlab_recon, unlab_z, unlab_mu, unlab_logvar = self.vae(unlabeled_imgs)
                transductive_loss = self.vae_loss(unlabeled_imgs,
                                                  unlab_recon, unlab_mu, unlab_logvar, self.beta)

                labeled_preds = self.discriminator(mu)
                unlabeled_preds = self.discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0)).to(device)
                unlab_real_preds = torch.ones(unlabeled_imgs.size(0)).to(device)

                # print(labeled_preds,unlabeled_preds,mu[0],unlabeled_imgs[0])
                # 我加的不然新的torch版本没法运行
                lab_real_preds = lab_real_preds.reshape(-1, 1)
                lab_real_preds = lab_real_preds.detach()
                unlab_real_preds = unlab_real_preds.reshape(-1, 1)
                unlab_real_preds = unlab_real_preds.detach()

                # print(torch.isnan(labeled_preds).int().sum().cpu())
                if torch.isnan(labeled_preds).int().sum() or torch.isnan(unlabeled_preds).int().sum():
                    print('NAN data')
                    continue

                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_real_preds)
                total_vae_loss = unsup_loss + transductive_loss + self.adversary_param * dsc_loss
                optim_vae.zero_grad()
                total_vae_loss.backward()
                optim_vae.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_vae_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)
                    labels = labels.to(device)

            # Discriminator step
            for count in range(self.num_adv_steps):
                with torch.no_grad():
                    _, _, mu, _ = self.vae(labeled_imgs)
                    _, _, unlab_mu, _ = self.vae(unlabeled_imgs)

                labeled_preds = self.discriminator(mu)
                unlabeled_preds = self.discriminator(unlab_mu)

                lab_real_preds = torch.ones(labeled_imgs.size(0))
                unlab_fake_preds = torch.zeros(unlabeled_imgs.size(0))

                lab_real_preds = lab_real_preds.to(device)
                unlab_fake_preds = unlab_fake_preds.to(device)
                lab_real_preds = lab_real_preds.reshape(-1, 1)
                lab_real_preds = lab_real_preds.detach()
                unlab_fake_preds = unlab_fake_preds.reshape(-1, 1)
                unlab_fake_preds = unlab_fake_preds.detach()

                if torch.isnan(labeled_preds).int().sum() or torch.isnan(unlabeled_preds).int().sum():
                    print('NAN data')
                    continue

                
                dsc_loss = self.bce_loss(labeled_preds, lab_real_preds) + \
                           self.bce_loss(unlabeled_preds, unlab_fake_preds)

                optim_discriminator.zero_grad()
                dsc_loss.backward()
                optim_discriminator.step()

                # sample new batch if needed to train the adversarial network
                if count < (self.num_adv_steps - 1):
                    labeled_imgs, _ = next(labeled_data)
                    unlabeled_imgs = next(unlabeled_data)

                    labeled_imgs = labeled_imgs.to(device)
                    unlabeled_imgs = unlabeled_imgs.to(device)
                    labels = labels.to(device)
                
            if batch_idx % 10 == 0:
                print('Current vae model loss: {:.4f}'.format(total_vae_loss.item()))
                print('Current discriminator model loss: {:.4f}'.format(dsc_loss.item()))

        return accFinal / len(loader_tr.dataset.X), total_vae_loss.item() + dsc_loss.item() + loss.item()

    def train(self, alpha=0, n_epoch=80):

        def weight_reset(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                m.reset_parameters()

        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        idxs_labeled = np.arange(self.n_pool)[self.idxs_lb]
        transform = self.args.transform_tr
        unlabeled_loader = DataLoader(self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                                            transform=transform), shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                               **self.args.loader_tr_args)

        labeled_loader = DataLoader(self.handler(self.X[idxs_labeled], torch.Tensor(self.Y.numpy()[idxs_labeled]).long(),transform=transform), 
                                    shuffle=True,
                                    pin_memory=True,
                                    # sampler = DistributedSampler(train_data),
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    **self.args.loader_tr_args)
        labeled_data = self.read_data(labeled_loader)
        unlabeled_data = self.read_data(unlabeled_loader, labels=False)

        if self.args.channels == 3:
            self.vae = VAE(self.args.img_size).to(device)
            self.discriminator = Discriminator(self.args.img_size).to(device)
        else:
            self.vae = VAE_MNIST().to(device)
            self.discriminator = Discriminator_MNIST().to(device)

        # self.vae = self.vae.apply(weight_reset)
        self.vae = nn.DataParallel(self.vae).to(device)
        # self.discriminator = self.discriminator.apply(weight_reset)
        self.discriminator = nn.DataParallel(self.discriminator).to(device)
        optim_vae = optim.Adam(self.vae.parameters(), lr=5e-4)
        optim_discriminator = optim.Adam(self.discriminator.parameters(), lr=5e-4)

        # optim_task_model = optim.SGD(task_model.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)
        

        self.clf = self.net.apply(weight_reset)
        self.clf = nn.DataParallel(self.clf).to(device)

        # optimizer = optim.Adam(self.clf.parameters(), lr=self.args.lr, weight_decay=0)
        optimizer = optim.SGD(self.clf.parameters(), lr = self.args.lr, weight_decay=5e-4, momentum=self.args.momentum)
        idxs_train = np.arange(self.n_pool)[self.idxs_lb]
        loader_tr = DataLoader(self.handler(self.X[idxs_train], torch.Tensor(self.Y.numpy()[idxs_train]).long(),
                                            transform=transform), shuffle=True,
                               **self.args.loader_tr_args)
        print(device)
        epoch = 0
        accCurrent = 0.
        lossOld = 0.
        recorder = RecorderMeter(n_epoch)
        while epoch < n_epoch:
            current_learning_rate, _ = adjust_learning_rate(optimizer, epoch, self.args.gammas, self.args.schedule, self.args)
            accCurrent, train_loss = self.vaal_train(epoch, loader_tr, optimizer, labeled_data, unlabeled_data, optim_vae, optim_discriminator)
            
            print(str(epoch) + ' training accuracy: ' + str(accCurrent), flush=True)
            test_acc = self.predict(self.X_te, self.Y_te)
            recorder.update(epoch, train_loss, accCurrent, 0, test_acc)
            epoch += 1
            # if abs(train_loss-lossOld) < 0.001:  # reset if not converging
            #     break
            # else: 
            #     lossOld = train_loss
        best_test_acc = recorder.max_accuracy(istrain=False)
        if self.args.save_model:
            self.save_model()
        return best_test_acc    

    def query(self, n):
        self.sampler = AdversarySampler(n)
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        unlabeled_loader = DataLoader(
            self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(),
                         transform=self.args.transform_te), shuffle=True,
            **self.args.loader_tr_args)
        querry_indices = self.sampler.sample(self.vae,
                                             self.discriminator,
                                             unlabeled_loader,
                                             )

        return idxs_unlabeled[querry_indices]

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
