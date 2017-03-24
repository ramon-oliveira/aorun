from collections import defaultdict
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.nn import MSELoss
from torch.autograd import Variable
from torch.optim import SGD

from . import losses
from . import optimizers
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
        assert not self.ready
        for prev_layer, next_layer in zip(self.layers[:-1], self.layers[1:]):
            next_layer.build(prev_layer.output_dim)
        self.ready = True

    def add(self, layer):
        self.layers.append(layer)
        self.ready = False

    def evaluate(self, X, y, metric, batch_size=32):
        n_samples, *_ = X.size()
        begin = min(batch_size, n_samples)
        end = n_samples + (n_samples % batch_size) + 1
        metric_sum = 0
        for split in range(begin, end, batch_size):
            X_batch = Variable(X[(split - batch_size):split], volatile=True)
            y_batch = Variable(y[(split - batch_size):split], volatile=True)

            out_batch = self.forward(X_batch)
            value = metric(y_batch, out_batch)
            metric_sum += value.data[0]

        return metric_sum / max(1, (end // batch_size))

    def forward(self, X):
        if not self.ready:
            self._build()
        y = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            y = layer.forward(y)
        return y

    def fit(self, X, y, loss, optimizer='adam', batch_size=32, epochs=10,
            shuffle=True, val_split=0.0, val_data=None, verbose=2):
        """
        verbose: 0, 1 or 2
            - 0 total silence
            - 1 only a progress bar for all training
            - 2 one progress bar for each epoch
        """
        X = utils.to_numpy(X)
        y = utils.to_numpy(y)

        if shuffle:
            X, y = utils.shuffle_arrays([X, y])

        if val_data is not None:
            X_val, y_val = val_data
        elif val_split > 0.0:
            (X, X_val), (y, y_val) = utils.split_arrays([X, y], val_split)
        else:
            X_val, y_val = None, None

        X = utils.to_tensor(X)
        y = utils.to_tensor(y)
        if X_val is not None:
            X_val = utils.to_tensor(X_val)
            y_val = utils.to_tensor(y_val)

        # create layers params if not created already
        if not self.ready:
            self._build()

        # set model params
        loss = losses.get(loss)
        optimizer = optimizers.get(optimizer)
        optimizer.params = self.params
        n_samples, *_ = X.size()
        batches = n_samples // batch_size + min(n_samples % batch_size, 1)
        self.loss = loss
        self.optimizer = optimizer
        self.batches = batches  # used by optmizers
        self.batch_size = batch_size

        progress_bar = None
        epochs_iterator = range(epochs)
        if verbose == 1:
            epochs_iterator = trange(epochs, desc=f'Epoch 0')
            progress_bar = epochs_iterator

        # batches
        begin = min(batch_size, n_samples)
        end = n_samples + (n_samples % batch_size) + 1
        step = batch_size
        history = defaultdict(list)
        for epoch in epochs_iterator:
            epoch_iterator = range(begin, end, step)
            if verbose == 2:
                progress_bar = tqdm(epoch_iterator)
            if progress_bar is not None:
                progress_bar.set_description(f'Epoch {epoch+1}')

            loss_sum = 0
            for batch, split in enumerate(epoch_iterator, start=1):
                X_batch = Variable(X[(split - batch_size):split])
                y_batch = Variable(y[(split - batch_size):split])

                out_batch = self.forward(X_batch)
                loss_value = loss(y_batch, out_batch)
                loss_value.backward()
                optimizer.step()
                loss_sum += loss_value.data[0]
                if progress_bar is not None:
                    progress_bar.set_postfix(loss=f'{loss_sum/batch:.4f}')
                    progress_bar.update()

            history['loss'].append(loss_sum / batches)
            if X_val is not None:
                val_loss = self.evaluate(X_val, y_val, loss, batch_size)
                history['val_loss'].append(val_loss)
                progress_bar.set_postfix(loss=f'{loss_sum/batches:.4f}',
                                         val_loss=f'{val_loss:.4f}')
                progress_bar.refresh()

            if verbose == 2:
                progress_bar.close()

        return history

    def predict(self, X):
        return_np = False
        if type(X) is np.ndarray:
            return_np = True
        y = self.forward(X)
        if return_np:
            return utils.to_numpy(y)
        else:
            return y
