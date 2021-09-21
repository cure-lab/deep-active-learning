import numpy as np
from .strategy import Strategy
from query_strategies.batchBALD import multi_bald
import dataclasses
import typing
from query_strategies.batchBALD.batchBALD import acquire_batch
from torch.utils.data import DataLoader
import torch
@dataclasses.dataclass
class AcquisitionBatch:
    indices: typing.List[int]
    scores: typing.List[float]
    orignal_scores: typing.Optional[typing.List[float]]

class BatchBALD(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BatchBALD, self).__init__(X, Y, idxs_lb, net, handler, args)
        self.net = net
        self.args = args

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        available_loader = DataLoader(self.handler(self.X[idxs_unlabeled], torch.Tensor(self.Y.numpy()[idxs_unlabeled]).long(), transform=self.args['transform']), shuffle=True, **self.args['loader_tr_args'])
        num_classes = 10
        num_inference_samples = 5
        available_sample_k = n # 我电脑上query很慢，后面看看原因
        min_candidates_per_acquired_item = 20
        min_remaining_percentage = 100
        initial_percentage = 100
        reduce_percentage = 0
        device = torch.device("cuda")
        model = self.net.bayesian_net #源代码里是这样的，model套一层sampler用来训练，query就直接用里面的net
        batch = acquire_batch(
                        bayesian_model=model,
                        available_loader=available_loader,
                        num_classes=num_classes,
                        k=num_inference_samples,
                        b=available_sample_k,
                        min_candidates_per_acquired_item=min_candidates_per_acquired_item,
                        min_remaining_percentage=min_remaining_percentage,
                        initial_percentage=initial_percentage,
                        reduce_percentage=reduce_percentage,
                        device=device,
                    )


        return idxs_unlabeled[batch.indices]
