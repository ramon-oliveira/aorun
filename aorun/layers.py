import torch
from torch.nn import Linear
from torch.nn import ReLU
from torch.autograd import Variable


class Layer(object):

    def __init__(self, input_dim=None):
        self.input_dim = input_dim

    @property
    def params(self):
        if self.l:
            return tuple(self.l.parameters())
        else:
            return tuple()


class Dense(Layer):

    def __init__(self, n, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.n = n
        self.output_dim = n
        if self.input_dim:
            self.l = Linear(self.input_dim, n)

    def build(self, input_dim):
        self.input_dim = input_dim
        self.l = Linear(self.input_dim, self.n)

    def forward(self, x):
        if type(x) is not Variable:
            x = Variable(x)
        return self.l(x)


class ProbabilisticDense(Layer):

    def __init__(self, n, *args, **kwargs):
        super(ProbabilisticDense, self).__init__(*args, **kwargs)
        self.n = n
        self.output_dim = n
        if self.input_dim:
            input_dim = self.input_dim
            output_dim = self.output_dim
            self.W_mu = Variable(torch.randn(input_dim, output_dim))
            self.W_sigma = Variable(torch.randn(input_dim, output_dim))
            self.bias = Variable(torch.randn(output_dim))

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W_mu = Variable(torch.randn(self.input_dim, self.output_dim))
        self.W_sigma = Variable(torch.randn(self.input_dim, self.output_dim))
        self.bias = Variable(torch.randn(self.output_dim))

    @property
    def params(self):
        return (self.W_mu, self.W_sigma, self.bias)

    def forward(self, x):
        if type(x) is not Variable:
            x = Variable(x)
        eps = Variable(torch.randn(self.input_dim, self.output_dim))
        W = self.W_mu + torch.log1p(torch.exp(self.W_sigma)) * eps

        xW = x @ W
        return xW + self.bias.expand_as(xW)


class Activation(Layer):

    def build(self, input_dim):
        self.output_dim = input_dim


class Relu(Activation):

    def __init__(self):
        self.l = ReLU()

    def forward(self, x):
        return self.l.forward(x)
