import pytest
from .context import aorun

import numpy as np
from torch import Tensor
from torch.autograd import Variable

from aorun import utils


def test_to_numpy():
    a = Tensor(10)
    a = utils.to_numpy(a)
    assert type(a) is np.ndarray

    a = Variable(Tensor(10))
    a = utils.to_numpy(a)
    assert type(a) is np.ndarray

    a = np.array([10])
    a = utils.to_numpy(a)
    assert type(a) is np.ndarray

    with pytest.raises(ValueError) as e:
        a = 'hahaha'
        utils.to_numpy(a)


def test_to_tensor():
    a = Tensor(10)
    a = utils.to_tensor(a)
    assert type(a) is Tensor

    a = Variable(Tensor(10))
    a = utils.to_tensor(a)
    assert type(a) is Variable

    a = np.array([10.0], dtype='float32')
    a = utils.to_tensor(a)
    assert type(a) is Tensor

    with pytest.raises(ValueError) as e:
        a = 'hahaha'
        utils.to_tensor(a)


def test_to_variable():
    a = Tensor(10)
    a = utils.to_variable(a)
    assert type(a) is Variable

    a = Variable(Tensor(10))
    a = utils.to_variable(a)
    assert type(a) is Variable

    a = np.array([10.0], dtype='float32')
    a = utils.to_variable(a)
    assert type(a) is Variable

    with pytest.raises(ValueError) as e:
        a = 'hahaha'
        utils.to_variable(a)
