from torch.nn import Linear
from torch.nn import ReLU
from torch.autograd import Variable


class Layer(object):

    def __init__(self, input_dim=None):
        self.input_dim = input_dim


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


class Activation(Layer):

    def build(self, input_dim):
        self.output_dim = input_dim


class Relu(Activation):

    def __init__(self):
        self.l = ReLU()

    def forward(self, x):
        return self.l.forward(x)
