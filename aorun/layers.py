import torch
from torch.nn import Linear
from torch.nn import ReLU
from torch.autograd import Variable
from torch.nn import Parameter


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
            self.W = Parameter(torch.randn(self.input_dim, self.output_dim))
            self.b = Parameter(torch.randn(self.output_dim))

    @property
    def params(self):
        return (self.W, self.b)

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W = Parameter(torch.randn(self.input_dim, self.output_dim))
        self.b = Parameter(torch.randn(self.output_dim))

    def forward(self, x):
        if type(x) is not Variable:
            x = Variable(x, requires_grad=False)
        xW = x @ self.W
        return xW + self.b.expand_as(xW)


class ProbabilisticDense(Layer):

    def __init__(self, n, *args, **kwargs):
        super(ProbabilisticDense, self).__init__(*args, **kwargs)
        self.n = n
        self.output_dim = n
        if self.input_dim:
            input_dim = self.input_dim
            output_dim = self.output_dim
            self.W_mu = Parameter(torch.randn(input_dim, output_dim))
            self.W_sigma = Parameter(torch.randn(input_dim, output_dim))
            self.b_mu = Parameter(torch.randn(output_dim))
            self.b_sigma = Parameter(torch.randn(output_dim))

    def build(self, input_dim):
        self.input_dim = input_dim
        self.W_mu = Parameter(torch.randn(self.input_dim, self.output_dim))
        self.W_sigma = Parameter(torch.randn(self.input_dim, self.output_dim))
        self.b_mu = Parameter(torch.randn(self.output_dim))
        self.b_sigma = Parameter(torch.randn(self.output_dim))

    @property
    def params(self):
        return (self.W_mu, self.W_sigma, self.b_mu, self.b_sigma)

    def forward(self, x):
        if type(x) is not Variable:
            x = Variable(x)

        W_eps = Variable(torch.randn(self.input_dim, self.output_dim))
        W = self.W_mu + torch.log1p(torch.exp(self.W_sigma)) * W_eps
        b_eps = Variable(torch.randn(self.output_dim))
        b = self.b_mu + torch.log1p(torch.exp(self.b_sigma)) * b_eps

        xW = x @ W
        return xW + b.expand_as(xW)


class Activation(Layer):

    def build(self, input_dim):
        self.output_dim = input_dim


class Relu(Activation):

    def __init__(self):
        self.l = ReLU()

    def forward(self, x):
        return self.l.forward(x)
