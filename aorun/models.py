from tqdm import tqdm, trange
import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD

from . import losses


class Model(object):

    def __init__(self, *layers):
        self.layers = list(layers)

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
        self.build()
        loss = losses.get(loss)
        optimizer.params = self.params
        n_samples, *_ = X.size()
        history = {'loss': []}
        begin = min(batch_size, n_samples)
        end = n_samples + (n_samples % batch_size) + 1
        step = batch_size

        for epoch in range(n_epochs):
            epoch_bar = trange(begin, end, step, desc=f'Epoch {epoch+1:2}')
            loss_sum = 0
            for ibatch, split in enumerate(epoch_bar, start=1):
                X_batch = Variable(X[(split - batch_size):split])
                y_batch = Variable(y[(split - batch_size):split])

                out_batch = self.forward(X_batch)
                loss_value = loss(y_batch, out_batch)
                loss_value.backward()
                optimizer.step()
                loss_sum += loss_value.data[0]
                epoch_bar.set_postfix(loss=f'{loss_sum/ibatch:.4f}')
            history['loss'].append(loss_value.data[0])

        return history
