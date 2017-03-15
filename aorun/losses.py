import torch


def mean_squared_error(true, pred):
    return torch.mean((true - pred)**2)


def binary_crossentropy(true, pred, eps=1e-9):
    p1 = true * torch.log(pred + eps)
    p2 = (1 - true) * torch.log(1 - pred + eps)
    return torch.mean(-(p1 + p2))


def categorical_crossentropy(true, pred, eps=1e-9):
    return torch.mean(-torch.sum(true * torch.log(pred + eps), dim=1))


# aliases short names
mse = mean_squared_error


def get(obj):
    if callable(obj):
        return obj
    elif type(obj) is str:
        if obj in globals():
            return globals()[obj]
        else:
            raise Exception(f'Unknown objective: {obj}')
    else:
        raise Exception('Objective must be a callable or str')
