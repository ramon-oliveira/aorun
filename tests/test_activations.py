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
    assert np.all(torch.sum(softmax(x), dim=1).data.numpy() == 1)


def test_get_softmax():
    x = Variable(torch.randn(10, 10))
    sm = torch.sum(activations.get('softmax')(x), dim=1)
    assert np.all(sm.data.numpy() == 1)


def test_layer_softmax():
    x = Variable(torch.randn(10, 10))
    l = Activation('softmax')
    assert np.all(torch.sum(l.forward(x), dim=1).data.numpy() == 1)
