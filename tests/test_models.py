import pytest
from .context import aorun

import numpy as np
import torch
from aorun.models import Model
from aorun.layers import Dense
from aorun.layers import Activation
from aorun.optimizers import SGD
from aorun.losses import mean_squared_error


def test_model_constructor_empty():
    model = Model()

    assert len(model.layers) == 0


def test_model_constructor_layers():
    model = Model(
        Dense(10),
        Activation('relu'),
        Dense(1)
    )

    assert len(model.layers) == 3
    assert type(model.layers[0]) == Dense
    assert type(model.layers[1]) == Activation


def test_model_add_layers():
    model = Model()
    model.add(Dense(10))
    model.add(Activation('relu'))
    model.add(Dense(1))

    assert len(model.layers) == 3
    assert type(model.layers[0]) == Dense
    assert type(model.layers[1]) == Activation


def test_model_forward():
    model = Model(
        Dense(10, input_dim=4),
        Dense(1),
        Dense(20)
    )

    x = torch.randn(2, 4)
    y = model.forward(x)

    assert y.size() == (2, 20)


def test_model_simple_fit():
    x = torch.rand(20, 4)
    y = torch.rand(20, 10)

    model = Model(
        Dense(10, input_dim=x.size()[-1]),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(y.size()[-1])
    )

    opt = SGD(lr=0.01, momentum=0.9)
    loss = mean_squared_error
    history = model.fit(x, y, loss=loss, optimizer='sgd', epochs=10, verbose=1)

    assert len(history['loss']) == 10
    assert all(type(v) is float for v in history['loss'])
    assert history['loss'] == sorted(history['loss'], reverse=True)


def test_model_fit_unknown_loss():
    x = torch.rand(20, 4)
    y = torch.rand(20, 10)

    model = Model(
        Dense(10, input_dim=x.size()[-1]),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(y.size()[-1])
    )

    assert len(model.params) > 0

    with pytest.raises(Exception) as e:
        model.fit(x, y, loss='UNKNOWN_TEST', batch_size=10, n_epoch=5)


def test_model_loss_str_param():
    x = torch.rand(20, 4)
    y = torch.rand(20, 10)

    model = Model(
        Dense(10, input_dim=x.size()[-1]),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(y.size()[-1])
    )

    opt = SGD(lr=0.01, momentum=0.9)

    history = model.fit(x, y, loss='mse', optimizer=opt, epochs=10)
    assert len(history['loss']) == 10
    assert all(type(v) is float for v in history['loss'])
    assert history['loss'] == sorted(history['loss'], reverse=True)

    loss = 'mean_squared_error'
    history = model.fit(x, y, loss=loss, optimizer=opt, epochs=10)
    assert len(history['loss']) == 10
    assert all(type(v) is float for v in history['loss'])
    assert history['loss'] == sorted(history['loss'], reverse=True)


def test_model_custom_loss():
    x = torch.rand(20, 4)
    y = torch.rand(20, 10)

    model = Model(
        Dense(10, input_dim=x.size()[-1]),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(y.size()[-1])
    )

    opt = SGD(lr=0.01, momentum=0.9)

    def mae(y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

    history = model.fit(x, y, loss=mae, optimizer=opt, epochs=10)
    assert len(history['loss']) == 10
    assert all(type(v) is float for v in history['loss'])
    assert history['loss'] == sorted(history['loss'], reverse=True)


def test_model_numpy_friendly():
    X = np.random.normal(size=[10, 10]).astype('float32')
    y = np.random.normal(size=[10, 1]).astype('float32')

    model = Model(
        Dense(10, input_dim=X.shape[-1]),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(y.shape[-1])
    )
    history = model.fit(X, y=y, loss='mse', optimizer='sgd', epochs=10)

    y_pred = model.forward(X)
    assert type(y_pred) is np.ndarray

    assert len(history['loss']) == 10
    assert all(type(v) is float for v in history['loss'])
    assert history['loss'] == sorted(history['loss'], reverse=True)


def test_model_adam_optmizer():
    X = np.random.normal(size=[10, 10]).astype('float32')
    y = np.random.normal(size=[10, 1]).astype('float32')

    model = Model(
        Dense(10, input_dim=X.shape[-1]),
        Activation('relu'),
        Dense(5),
        Activation('relu'),
        Dense(y.shape[-1])
    )
    history = model.fit(X, y=y, loss='mse', optimizer='adam', epochs=10)

    y_pred = model.forward(X)
    assert type(y_pred) is np.ndarray

    assert len(history['loss']) == 10
    assert all(type(v) is float for v in history['loss'])
    assert history['loss'] == sorted(history['loss'], reverse=True)
