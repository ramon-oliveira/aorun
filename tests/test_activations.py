from .context import aorun

from aorun import activations
from aorun.activations import relu
from aorun.layers import Activation

import numpy as np
import torch
from torch.autograd import Variable


def test_relu():
    x = Variable(torch.randn(10, 10))
    assert np.any(x.data.numpy() < 0.0)
    assert np.any(relu(x).data.numpy() >= 0.0)


def test_get_relu():
    x = Variable(torch.randn(10, 10))
    assert np.any(x.data.numpy() < 0.0)
    assert np.any(activations.get('relu')(x).data.numpy() >= 0.0)


def test_relu_layer():
    x = Variable(torch.randn(10, 10))
    l = Activation('relu')
    assert np.any(x.data.numpy() < 0.0)
    assert np.any(l.forward(x).data.numpy() >= 0.0)
