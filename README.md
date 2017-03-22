# Aorun: A high-level API for PyTorch

Aorun is highly inspired by Keras API.

## Getting started

Here is a simple regression example:

```python
from aorun.models import Model
from aorun.layers import Dense, Activation

model = Model()
model.add(Dense(10, input_dim=3))
model.add(Activation('relu'))
model.add(Dense(1))

model.fit(X_train, y_train, loss='mse', optimizer='sgd')

y_pred = model.forward(X_test)
```

## Why Aorun?

[Aorun](https://en.wikipedia.org/wiki/Aorun) is a Dinosaur. Dinosaurs are cool.
