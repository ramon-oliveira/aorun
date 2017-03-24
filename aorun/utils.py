import torch
from torch import Tensor
from torch.autograd import Variable
import numpy as np
from functools import wraps


def shuffle_arrays(arrays):
    # arrays must have the same length
    assert np.all(np.array([len(a) for a in arrays]) == len(arrays[0]))
    idxs = np.arange(len(arrays[0]))
    np.random.shuffle(idxs)
    return [a[idxs] for a in arrays]


def split_arrays(arrays, proportion):
    """
    proportion will be in the last part
    examples:
        proportion = 0.7
        [30%] | [70%]
        proportion = 0.3
        [70%] | [30%]
    """
    # arrays must have the same length
    assert np.all(np.array([len(a) for a in arrays]) == len(arrays[0]))
    proportion = 1 - proportion
    split = int(len(arrays[0]) * proportion)
    return [(a[:split], a[split:]) for a in arrays]


def to_tensor(a):
    if type(a) is np.ndarray:
        return torch.from_numpy(a)
    elif type(a) is Tensor or type(a) is Variable:
        return a
    else:
        raise ValueError('Unknown value type: {0}'.format(type(a)))


def to_variable(a):
    a = to_tensor(a)
    if type(a) is Variable:
        return a
    else:
        return Variable(a)


def to_numpy(a):
    if type(a) is Tensor:
        return a.numpy()
    elif type(a) is Variable:
        return a.data.numpy()
    elif type(a) is np.ndarray:
        return a
    else:
        raise ValueError('Unknown value type: {0}'.format(type(a)))
