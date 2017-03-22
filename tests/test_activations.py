import pytest
from .context import aorun

from aorun import activations
from aorun.activations import relu
from aorun.activations import softmax
from aorun.layers import Activation

import numpy as np
import torch
from torch.autograd import Variable


def test_get_unknown():
    with pytest.raises(Exception) as e:
        activations.get('UNKNOWN_TEST')

    with pytest.raises(Exception) as e:
        activations.get(23452345)


def test_custom_activation():
    def custom(x):
        return x

    activation = activations.get(custom)
    x = torch.randn(10)
    assert torch.equal(x, custom(x))


def test_relu():
    x = Variable(torch.randn(10, 10))
    assert np.any(x.data.numpy() < 0.0)
    assert np.any(relu(x).data.numpy() >= 0.0)


def test_get_relu():
    x = Variable(torch.randn(10, 10))
    assert np.any(x.data.numpy() < 0.0)
    assert np.any(activations.get('relu')(x).data.numpy() >= 0.0)


def test_layer_relu():
    x = Variable(torch.randn(10, 10))
    l = Activation('relu')
    assert np.any(x.data.numpy() < 0.0)
    assert np.any(l.forward(x).data.numpy() >= 0.0)


def test_softmax():
    x = Variable(torch.randn(10, 10))
    sum_softmax_x = torch.sum(softmax(x), dim=1).data.numpy()
    assert np.all(np.abs(sum_softmax_x - 1) <= 1e-6)


def test_get_softmax():
    x = Variable(torch.randn(10, 10))
    softmax = activations.get('softmax')
    sum_softmax_x = torch.sum(softmax(x), dim=1).data.numpy()
    assert np.all(np.abs(sum_softmax_x - 1) <= 1e-6)


def test_layer_softmax():
    x = Variable(torch.randn(10, 10))
    l = Activation('softmax')
    sum_softmax_x = torch.sum(l.forward(x), dim=1).data.numpy()
    assert np.all(np.abs(sum_softmax_x - 1) <= 1e-6)
