import numpy as np
import torch
from torch import Tensor
from torch.autograd import Variable
from .layers import ProbabilisticDense


def log_gaussian(x, mu, sigma):
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


def variational_loss(model, log_likelihood):
    log_likelihood = get(log_likelihood)
    prior_ratio = 0.5
    prior_mu = Variable(Tensor([0]))
    prior_sigma1 = Variable(Tensor([2.0]))
    prior_sigma2 = Variable(Tensor([0.05]))

    def loss(true, pred):
        log_p = Variable(torch.Tensor([0.0]))
        log_q = Variable(torch.Tensor([0.0]))
        for layer in model.layers:
            if type(layer) is ProbabilisticDense:
                # posterior
                x = layer.W
                mu = layer.W_mu
                sigma = torch.log1p(torch.exp(layer.W_rho))
                log_q += log_gaussian(x, mu, sigma).sum()

                # prior
                mu = prior_mu.expand_as(x)
                sigma1 = prior_sigma1.expand_as(x)
                sigma2 = prior_sigma2.expand_as(x)
                p1 = prior_ratio * log_gaussian(x, mu, sigma1)
                p2 = (1 - prior_ratio) * log_gaussian(x, mu, sigma2)
                log_p += torch.sum(p1 + p2)

        ll = -log_likelihood(true, pred)
        return (log_q - log_p) / model.batches - ll
    return loss

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
