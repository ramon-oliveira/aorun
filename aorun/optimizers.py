import torch
from torch.autograd import Variable
from torch.optim import Adam


class Optimizer(object):

    def clear_gradients(self):
        if self.params:
            for p in self.params:
                p.grad.data.zero_()


class SGD(Optimizer):

    def __init__(self, params=None, lr=0.001, decay=0.0, momentum=0.0):
        self.params = params
        self.lr = lr
        self.decay = decay
        self.momentum = momentum
        self.updates = []

    def step(self, clear=True):
        if self.params is None:
            raise Exception('None parameters')

        if len(self.updates) == 0:
            for p in self.params:
                update = torch.zeros(p.size())
                self.updates.append(update)

        for i, (p, update) in enumerate(zip(self.params, self.updates)):
            cur_update = self.momentum * update + self.lr * p.grad.data
            p.data.sub_(cur_update)
            self.updates[i] = cur_update

        self.lr = max(1e-9, self.lr - self.decay)

        if clear:
            self.clear_gradients()
