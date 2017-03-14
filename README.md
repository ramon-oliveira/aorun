# Aorun: A keras-like wrapper for Pytorch

## Getting started

Here is a simple regression example:

```python
from aorun.models import Model
from aorun.layers import Dense, Relu
from aorun.optimizers import SGD

model = Model()
model.add(Dense(10, input_dim=3))
model.add(Relu())
model.add(Dense(1))

sgd = SGD(lr=0.001)
model.fit(X_train, y_train, loss='mse', optimizer=sgd)

y_pred = model.forward(X_test)
```

## TODO:

* Numpy friendly
* Add layers:
    - Convolutional2D
    - RNN
    - GRU
    - LSTM
* Add objectives:
    - mean absolute error
    - binary crossentropy
    - categoriacal crossentropy
    - KL divergence
* Add optimizers:
    - Adam
    - RMSProp
* Add examples:
    - mnist
    - cifar10
* Bug fix:
    - Stabilize training in probabilistic layers

## Why Aorun?

[Aorun](https://en.wikipedia.org/wiki/Aorun) is a Dinosaur. Dinosaurs are cool.
