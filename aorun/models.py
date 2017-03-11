

class Model(object):

    def __init__(self, *layers):
        self.layers = list(layers)
        self.build()

    def build(self):
        for prev_layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            next_layer.build(prev_layer.output_dim)

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, x):
        y = self.layers[0].forward(x)
        for layer in self.layers[1:]:
            y = layer.forward(y)
        return y
