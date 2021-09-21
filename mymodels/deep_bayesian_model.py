from torch import nn as nn
from torch.nn import functional as F
from torch import Tensor
import torch
import mc_dropout
import gc
import math
eval_bayesian_model_consistent_cuda_chunk_size = 1024
sampler_model_cuda_chunk_size = 1024

class SamplerModel(nn.Module):
    def __init__(self, bayesian_net: mc_dropout.BayesianModule):
        super().__init__()
        self.bayesian_net = bayesian_net
        self.num_classes = bayesian_net.num_classes

    def forward(self, input: torch.Tensor):
        global sampler_model_cuda_chunk_size
        if self.training:
            self.k = 1
        else:
            self.k = 100

        if self.training:
            result = self.bayesian_net(input, self.k)
            return logit_mean(result, dim=1, keepdim=False), result
        else:
            mc_output_B_C = torch.zeros((input.shape[0], self.num_classes), dtype=torch.float64, device=input.device)

            k = self.k

            chunk_size = sampler_model_cuda_chunk_size if input.device.type == "cuda" else 32

            k_lower = 0
            while k_lower < k:
                try:
                    k_upper = min(k_lower + chunk_size, k)
                    # Reset the mask all around.
                    self.bayesian_net.eval()

                    mc_output_B_K_C = self.bayesian_net(input, k_upper - k_lower)
                except RuntimeError as exception:
                    if is_cuda_out_of_memory(exception) or is_cudnn_snafu(exception):
                        chunk_size //= 2
                        if chunk_size <= 0:
                            raise
                        if sampler_model_cuda_chunk_size != chunk_size:
                            print(f"New sampler_model_cuda_chunk_size={chunk_size} ({exception})")
                            sampler_model_cuda_chunk_size = chunk_size

                        gc.collect()
                        torch.cuda.empty_cache()
                else:
                    mc_output_B_C += torch.sum(mc_output_B_K_C.double().exp_(), dim=1, keepdim=False)
                    k_lower += chunk_size

            return (mc_output_B_C / k).log_(), mc_output_B_C

class BayesianNet(mc_dropout.BayesianModule):
    def __init__(self, num_classes):
        super().__init__(num_classes)

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv1_drop = mc_dropout.MCDropout2d()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = mc_dropout.MCDropout2d()
        self.fc1 = nn.Linear(1024, 128)
        self.fc1_drop = mc_dropout.MCDropout()
        self.fc2 = nn.Linear(128, num_classes)

    def mc_forward_impl(self, input: Tensor):
        input = F.relu(F.max_pool2d(self.conv1_drop(self.conv1(input)), 2))
        input = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(input)), 2))
        input = input.view(-1, 1024)
        input = F.relu(self.fc1_drop(self.fc1(input)))
        input = self.fc2(input)
        input = F.log_softmax(input, dim=1)

        return input

def is_cuda_out_of_memory(exception):
    return (
        isinstance(exception, RuntimeError) and len(exception.args) == 1 and "CUDA out of memory." in exception.args[0]
    )

def is_cudnn_snafu(exception):
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )

def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])