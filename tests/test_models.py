from .context import aorun

from aorun.models import Model
from aorun.layers import Dense
from aorun.layers import Relu

import torch


def test_model_constructor_empty():

    model = Model()

    assert len(model.layers) == 0


def test_model_constructor_layers():

    model = Model(
        Dense(10),
        Relu(),
        Dense(1)
    )

    assert len(model.layers) == 3
    assert type(model.layers[0]) == Dense
    assert type(model.layers[1]) == Relu


def test_model_add_layers():

    model = Model()
    model.add(Dense(10))
    model.add(Relu())
    model.add(Dense(1))

    assert len(model.layers) == 3
    assert type(model.layers[0]) == Dense
    assert type(model.layers[1]) == Relu


def test_model_forward():

    model = Model(
        Dense(10, input_dim=4),
        Dense(1),
        Dense(20)
    )

    x = torch.Tensor(2, 4)
    y = model.forward(x)

    assert y.size() == (2, 20)
