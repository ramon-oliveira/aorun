from .context import aorun

import torch
from torch.autograd import Variable
from aorun.objectives import mean_squared_error
from aorun.objectives import binary_crossentropy
from aorun.objectives import categorical_crossentropy


def test_mse_variable():
    true = Variable(torch.Tensor([11, 11]))
    pred = Variable(torch.Tensor([10, 10]))

    loss = mean_squared_error(true, pred)

    assert type(loss) is Variable
    assert loss == 1


def test_mse_tensor():
    true = torch.Tensor([11, 11])
    pred = torch.Tensor([10, 10])

    loss = mean_squared_error(true, pred)

    assert loss == 1


def test_binary_crossentropy():
    true = torch.Tensor([1, 1, 1])

    loss = binary_crossentropy(true, true)
    assert loss == 0

    pred = torch.Tensor([[0, 0, 0]])
    loss = binary_crossentropy(true, pred)
    assert loss > 1


def test_categorical_crossentropy():
    true = torch.Tensor([[1, 0], [0, 1], [1, 0]])

    loss = categorical_crossentropy(true, true)
    assert loss == 0

    pred = torch.Tensor([[0, 1], [0, 1], [1, 0]])
    loss = categorical_crossentropy(true, pred)
    assert loss > 1
