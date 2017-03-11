from tqdm import tqdm
import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD


class Model(object):

    def __init__(self, *layers):
        self.layers = list(layers)
        self.build()

    @property
    def params(self):
        return [p for layer in self.layers for p in layer.params]

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

    def fit(self, X, y, loss, optimizer, batch_size=32, n_epochs=10):
        optimizer.params = self.params
        n_samples, *_ = X.size()
        history = {'loss': []}
        begin = min(batch_size, n_samples)
        end = n_samples + (n_samples % batch_size) + 1
        for epoch in range(n_epochs):
            for split in tqdm(range(begin, end, batch_size)):
                X_batch = Variable(X[(split - batch_size):split])
                y_batch = Variable(y[(split - batch_size):split])

                out_batch = self.forward(X_batch)
                loss_value = loss(y_batch, out_batch)
                loss_value.backward()
                optimizer.step()
            history['loss'].append(loss_value.data[0])

        return history
