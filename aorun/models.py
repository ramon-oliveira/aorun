

class Model(object):

    def __init__(self, *layers):
        self.layers = list(layers)

    def add(self, layer):
        self.layers.append(layer)
