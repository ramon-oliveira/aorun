import torch
from torch.autograd import Variable


class Optimizer(object):

    def __init__(self, params=None, decay=0.0, epsilon=1e-8):
        self.params = params
        self.decay = decay
        self.epsilon = epsilon

    def clear_gradients(self):
        if self.params is not None:
            for p in self.params:
                p.grad.data.zero_()

    def step(self):
        if self.params is None:
            raise Exception('None parameters')


class SGD(Optimizer):

    def __init__(self, lr=0.001, momentum=0.0, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)
        self.lr = lr
        self.momentum = momentum
        self.updates = []

    def step(self):
        super(SGD, self).step()

        if len(self.updates) == 0:
            for p in self.params:
                update = torch.zeros(p.size())
                self.updates.append(update)

        for i, (p, update) in enumerate(zip(self.params, self.updates)):
            cur_update = self.momentum * update + self.lr * p.grad.data
            p.data.sub_(cur_update)
            self.updates[i] = cur_update

        self.lr = max(1e-9, self.lr - self.decay)
        self.clear_gradients()


class Adam(Optimizer):

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, *args, **kwargs):
        super(Adam, self).__init__(*args, **kwargs)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        # average gradients
        self.m = {}
        # average gradients**2
        self.v = {}
        # timestep
        self.t = 0

    def step(self):
        super(Adam, self).step()
        self.t += 1

        if len(self.m) == 0:
            for p in self.params:
                self.m[p] = torch.zeros(p.size())
                self.v[p] = torch.zeros(p.size())

        for p in self.params:
            mt = self.beta1 * self.m[p] + (1 - self.beta1) * p.grad.data
            vt = self.beta2 * self.v[p] + (1 - self.beta2) * p.grad.data**2
            m = mt / (1 - self.beta1**self.t)
            v = vt / (1 - self.beta2**self.t)

            rate = self.lr / (torch.sqrt(v) + self.epsilon)
            p.data.sub_(rate * m)

            self.m[p] = mt
            self.v[p] = vt

        self.clear_gradients()

# Alias
sgd = SGD
adam = Adam


def get(obj):
    if hasattr(obj, 'step'):
        return obj
    elif type(obj) is str:
        if obj in globals():
            return globals()[obj]()
        else:
            raise Exception(f'Unknown optmizer: {obj}')
    else:
        raise Exception('Optimizer must be a callable or str')
