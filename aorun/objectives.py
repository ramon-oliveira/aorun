import torch


def mean_squared_error(true, pred):
    return torch.mean((true - pred)**2)
