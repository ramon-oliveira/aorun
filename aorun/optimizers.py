from torch.optim import SGD as SGD_torch


class Optimizer(object):
    pass


class SGD(Optimizer):

    def __init__(self, params=None, lr=0.001, momentum=0.0, nesterov=False):
        self.params = params
        self.lr = lr
        self.momentum = momentum
        self.nesterov = nesterov

    def clear_gradients(self):
        if self.params:
            for p in self.params:
                p.grad.data.zero_()

    def step(self, clear=True):
        if self.params is None:
            raise Exception('None parameters')

        for p in self.params:
            p.data.sub_(self.lr * p.grad.data)

        if clear:
            self.clear_gradients()
