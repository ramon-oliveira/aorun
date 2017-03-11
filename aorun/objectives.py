import torch


def mean_squared_error(true, pred):
    return torch.mean((true - pred)**2)

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
