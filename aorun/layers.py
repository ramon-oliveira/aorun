import torch
from torch.autograd import Variable
from torch.nn import Parameter

from . import activations
from . import initializers


class Layer(object):

    def __init__(self, input_dim=None):
        self.input_dim = input_dim


class Dense(Layer):

    def __init__(self, units, init='glorot_uniform', *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.units = units
        self.output_dim = units
        self.init = initializers.get(init)
        if self.input_dim:
            self.build(self.input_dim)

    @property
    def params(self):
        return (self.W, self.b)

    def build(self, input_dim):
        self.input_dim = input_dim
        W_shape = [self.input_dim, self.output_dim]
        b_shape = [self.output_dim]
        self.W = self.init(W_shape, self.input_dim, self.output_dim)
        self.b = self.init(b_shape, self.input_dim, self.output_dim)

    def forward(self, x):
        if type(x) is not Variable:
            x = Variable(x)
        xW = x @ self.W
        return xW + self.b.expand_as(xW)


class ProbabilisticDense(Layer):

    def __init__(self, units, init='glorot_uniform', *args, **kwargs):
        super(ProbabilisticDense, self).__init__(*args, **kwargs)
        self.units = units
        self.output_dim = units
        self.init = initializers.get(init)
        if self.input_dim:
            self.build(self.input_dim)

    def build(self, input_dim):
        self.input_dim = input_dim
        W_shape = [self.input_dim, self.output_dim]
        b_shape = [self.output_dim]
        self.W_mu = self.init(W_shape, self.input_dim, self.output_dim)
        self.W_rho = self.init(W_shape, self.input_dim, self.output_dim)
        self.b_mu = self.init(b_shape, self.input_dim, self.output_dim)
        self.b_rho = self.init(b_shape, self.input_dim, self.output_dim)

    @property
    def params(self):
        return (self.W_mu, self.W_rho, self.b_mu, self.b_rho)

    def forward(self, X):
        if type(X) is not Variable:
            X = Variable(X)

        W_eps = Variable(torch.randn(self.input_dim, self.output_dim))
        self.W = W = self.W_mu + torch.log1p(torch.exp(self.W_rho)) * W_eps
        b_eps = Variable(torch.randn(self.output_dim))
        self.b = b = self.b_mu + torch.log1p(torch.exp(self.b_rho)) * b_eps
        XW = X @ W
        return XW + b.expand_as(XW)


class Activation(Layer):

    def __init__(self, activation):
        self.activation = activations.get(activation)

    @property
    def params(self):
        return tuple()

    def build(self, input_dim):
        self.output_dim = input_dim

    def forward(self, x):
        return self.activation(x)
