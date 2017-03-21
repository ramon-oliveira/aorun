from tqdm import tqdm, trange
import numpy as np
import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD

from . import losses
from . import utils


class Model(object):

    def __init__(self, *layers):
        self.layers = list(layers)
        self.ready = False

    @property
    def params(self):
        if not self.ready:
            self._build()
        return [p for layer in self.layers for p in layer.params]

    def _build(self):
        for prev_layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            next_layer.build(prev_layer.output_dim)
        self.ready = True

    def add(self, layer):
        self.layers.append(layer)
        self.ready = False

    @utils.numpyio
    def forward(self, X):
        if not self.ready:
            self._build()
        y = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            y = layer.forward(y)
        return y

    @utils.numpyio
    def fit(self, X, y, loss, optimizer, batch_size=32, epochs=10, verbose=2):
        if not self.ready:
            self._build()
        loss = losses.get(loss)
        optimizer.params = self.params
        n_samples, *_ = X.size()
        history = {'loss': []}
        begin = min(batch_size, n_samples)
        end = n_samples + (n_samples % batch_size) + 1
        step = batch_size
        self.batches = n_samples // batch_size + n_samples % batch_size
        self.batch_size = batch_size

        bar = None
        epochs_iterator = range(epochs)
        if verbose == 1:
            epochs_iterator = trange(epochs, desc=f'Epoch 0')
            bar = epochs_iterator

        for epoch in epochs_iterator:
            epoch_iterator = range(begin, end, step)
            if verbose == 2:
                epoch_iterator = tqdm(epoch_iterator, desc=f'Epoch {epoch+1}')
                bar = epoch_iterator

            loss_sum = 0
            for ibatch, split in enumerate(epoch_iterator, start=1):
                X_batch = Variable(X[(split - batch_size):split])
                y_batch = Variable(y[(split - batch_size):split])

                out_batch = self.forward(X_batch)
                loss_value = loss(y_batch, out_batch)
                loss_value.backward()
                optimizer.step()
                loss_sum += loss_value.data[0]
                if bar:
                    bar.set_postfix(loss=f'{loss_sum/ibatch:.4f}')
                    bar.set_description(f'Epoch {epoch+1}')
            history['loss'].append(loss_value.data[0])

        return history
