import numpy as np
import torch
import torch.nn.functional as F
from .strategy import Strategy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AdversarialBIM(Strategy):
	def __init__(self, X, Y, X_te, Y_te, idxs_lb, net, handler, args, eps=0.05):
		super(AdversarialBIM, self).__init__(X, Y, X_te, Y_te, idxs_lb, net, handler, args)
		self.eps = eps
		
	def cal_dis(self, x):
		# x = x.to(device=device)
		nx = torch.unsqueeze(x, 0)
		nx.requires_grad_()
		eta = torch.zeros(nx.shape)

		out, e1 = self.clf((nx+eta).to(self.device))
		py = out.max(1)[1]
		ny = out.max(1)[1]
		while py.item() == ny.item():
			loss = F.cross_entropy(out, ny)
			loss.backward()

			eta += self.eps * torch.sign(nx.grad.data)
			nx.grad.data.zero_()

			out, e1 = self.clf((nx+eta).to(self.device))
			py = out.max(1)[1]

		return (eta*eta).sum()

	def query(self, n):
		idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]

		# self.clf = self.clf.to('cpu')
		if not self.pretrained:
			self.clf.eval()
		else:
			self.clf.classifier.eval()
		dis = np.zeros(idxs_unlabeled.shape)
		transform = self.args.transform_te if not self.pretrained else self.preprocessing
		data_pool = self.handler(self.X[idxs_unlabeled], self.Y[idxs_unlabeled], 
						transform=transform)
		for i in range(len(idxs_unlabeled)):
			if i % 100 == 0:
				print('adv {}/{}'.format(i, len(idxs_unlabeled)))
			x, y, idx = data_pool[i]
			dis[i] = self.cal_dis(x)

		# self.clf = self.clf.to(self.device)

		return idxs_unlabeled[dis.argsort()[:n]]


