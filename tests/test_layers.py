import pytest
from .context import aorun

import torch
from aorun.layers import Dense
from aorun.layers import ProbabilisticDense
from aorun.layers import Conv2D
from aorun.layers import Activation


def test_dense_layer_output_dim():
    l = Dense(5, input_dim=10)

    assert l.output_dim == 5


def test_dense_layer_forward():
    x = torch.randn(2, 10)
    l = Dense(5, input_dim=10)
    y = l.forward(x)

    assert y.size() == (2, 5)


def test_dense_multiple_layers():
    x = torch.randn(2, 10)
    l1 = Dense(5, input_dim=10)
    l2 = Dense(3, input_dim=5)

    y = l1.forward(x)
    assert y.size() == (2, 5)

    y = l2.forward(y)
    assert y.size() == (2, 3)


def test_relu_output_size():
    x = torch.randn(2, 2)
    l1 = Dense(3, input_dim=2)
    l2 = Activation('relu')

    y = l1.forward(x)
    y = l2.forward(y)

    assert y.size() == (2, 3)
    assert (y.data >= 0).sum() == 6


def test_layer_get_params():
    l = Dense(3, input_dim=3)
    assert len(l.params) == 2

    l = Activation('relu')
    assert len(l.params) == 0


def test_layer_probabilistic_dense():
    x = torch.randn(2, 10)
    l = ProbabilisticDense(5, input_dim=10)

    y1 = l.forward(x)
    assert y1.size() == (2, 5)

    y2 = l.forward(x)
    assert y2.size() == (2, 5)
    assert not torch.equal(y1.data, y2.data)


def test_layer_probabilistic_dense_build():
    x = torch.randn(2, 10)
    l = ProbabilisticDense(5)
    l.build(10)

    y1 = l.forward(x)
    assert y1.size() == (2, 5)

    y2 = l.forward(x)
    assert y2.size() == (2, 5)
    assert not torch.equal(y1.data, y2.data)


def test_layer_conv2d():
    x = torch.randn(2, 3, 9, 9)
    layer = Conv2D(64, (3, 3), input_dim=[3, 9, 9])

    y1 = layer.forward(x)
    assert y1.size() == (2, 64, 7, 7)


def test_layer_conv2d_params():
    x = torch.randn(2, 3, 9, 9)
    layer = Conv2D(64, (3, 3), input_dim=[3, 9, 9])

    assert len(layer.params) == 2
