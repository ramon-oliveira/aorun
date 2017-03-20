import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from .layers import ProbabilisticDense


def _log_gaussian(x, mu, sigma):
    assert x.size() == mu.size() == sigma.size()

    log_sigma = torch.log(sigma)
    # log(2 * pi) == 1.8378770664093453
    log2pi_2 = Variable(Tensor([1.8378770664093453 / 2]))
    log2pi_2 = log2pi_2.expand_as(mu)

    return -log_sigma - log2pi_2 - (x - mu)**2 / (2 * sigma**2)


def mean_squared_error(true, pred):
    return ((true - pred)**2).mean()


def binary_crossentropy(true, pred, eps=1e-9):
    p1 = true * torch.log(pred + eps)
    p2 = (1 - true) * torch.log(1 - pred + eps)
    return torch.mean(-(p1 + p2))


def categorical_crossentropy(true, pred, eps=1e-9):
    return torch.mean(-torch.sum(true * torch.log(pred + eps), dim=1))


def variational_loss(true, pred, model, log_likelihood):
    log_p = Variable(torch.Tensor([0]))
    log_q = Variable(torch.Tensor([0]))
    for layer in model.layers:
        if type(layer) is ProbabilisticDense:
            # posterior
            x = layer.W
            mu = layer.W_mu
            sigma = torch.log1p(torch.exp(layer.W_rho))
            log_q += _log_gaussian(x, mu, sigma).sum()

            # prior
            mu = Variable(Tensor([0])).expand_as(x)
            sigma = Variable(Tensor([0.05])).expand_as(x)
            log_p += _log_gaussian(x, mu, sigma).sum()

    ll = log_likelihood(true, pred)

    return -ll - log_q - log_p


# aliases short names
mse = mean_squared_error


def get(obj):
    if callable(obj):
        return obj
    elif type(obj) is str:
        if obj in globals():
            return globals()[obj]
        else:
            raise Exception(f'Unknown loss: {obj}')
    else:
        raise Exception('Loss must be a callable or str')
