import numpy as np
import torch
from torch.nn import Parameter


def glorot_uniform(shape, in_units, out_units):
    limit = np.sqrt(6 / (in_units + out_units))
    W = np.random.uniform(-limit, limit, size=shape).astype('float32')
    W = torch.from_numpy(W)
    return Parameter(W)


def get(obj):
    if callable(obj):
        return obj
    elif type(obj) is str:
        if obj in globals():
            return globals()[obj]
        else:
            raise Exception(f'Unknown initializer: {obj}')
    else:
        raise Exception('Initializer must be a callable or str')
