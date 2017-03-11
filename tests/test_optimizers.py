from .context import aorun

import torch
from torch.autograd import Variable
from aorun.optimizers import SGD


def test_sgd_learning_rate():
    X = Variable(torch.rand(5, 3))
    y = Variable(torch.rand(5))
    w = Variable(torch.rand(3), requires_grad=True)
    opt = SGD([w], lr=0.1)

    o = X @ w
    loss = torch.sum((y - o)**2)
    loss.backward()
    opt.step()

    assert loss.data[0] > torch.sum((y - X @ w)**2).data[0]
