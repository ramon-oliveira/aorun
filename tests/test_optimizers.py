import pytest
from .context import aorun

import torch
from torch.autograd import Variable
from aorun.losses import mean_squared_error
from aorun import optimizers
from aorun.optimizers import SGD


def test_optmizers_get():
    with pytest.raises(Exception) as e:
        optimizers.get('UNKNOWN_TEST')

    with pytest.raises(Exception) as e:
        optimizers.get(123123)


def test_sgd_without_params():
    opt = optimizers.get('sgd')
    with pytest.raises(Exception) as e:
        opt.step()


def test_sgd_learning_rate():
    X = Variable(torch.rand(5, 3))
    y = Variable(torch.rand(5))
    w = Variable(torch.rand(3), requires_grad=True)
    opt = SGD([w], lr=0.1)

    o = X @ w
    loss = mean_squared_error(y, o)
    loss.backward()
    opt.step()

    assert loss > mean_squared_error(y, X @ w)


def test_sgd_momentum():
    X = Variable(torch.rand(5, 3))
    y = Variable(torch.rand(5))
    w = Variable(torch.rand(3), requires_grad=True)
    opt = SGD([w], lr=0.1, momentum=0.99)

    o = X @ w
    loss = mean_squared_error(y, o)
    loss.backward()
    opt.step()

    assert loss > mean_squared_error(y, X @ w)
