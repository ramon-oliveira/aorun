from .context import aorun

import torch
from torch.autograd import Variable
from aorun.objectives import mean_squared_error


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
