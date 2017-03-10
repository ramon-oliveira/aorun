

class Layer(object):
    pass


class Dense(Layer):

    def __init__(self, n, *args, **kwargs):
        super(Dense, self).__init__(*args, **kwargs)
        self.n = n

    def __len__(self):
        return self.n


class ProbabilisticDense(Dense):

    def __init__(self, *args, **kwargs):
        super(ProbabilisticDense, self).__init__(*args, **kwargs)


class Activation(Layer):
    pass


class Relu(Activation):
    pass
