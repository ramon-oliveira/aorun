from .context import aorun

import torch
from torch.autograd import Variable
from aorun.objectives import mean_squared_error
from aorun.optimizers import SGD


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
