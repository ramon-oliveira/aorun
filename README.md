# Aorun: Deep Learning over PyTorch

[![build](https://travis-ci.org/ramon-oliveira/aorun.svg?branch=master)](https://travis-ci.org/ramon-oliveira/aorun)
[![coverage](https://coveralls.io/repos/github/ramon-oliveira/aorun/badge.svg)](https://coveralls.io/github/ramon-oliveira/aorun)
[![Code Climate](https://codeclimate.com/github/ramon-oliveira/aorun/badges/gpa.svg)](https://codeclimate.com/github/ramon-oliveira/aorun)
[![python](https://img.shields.io/pypi/pyversions/aorun.svg)](https://pypi.python.org/pypi/aorun)
[![license](https://img.shields.io/github/license/ramon-oliveira/aorun.svg)](https://github.com/ramon-oliveira/aorun/blob/master/LICENSE)

Aorun intend to be a [Keras](https://keras.io) with PyTorch as backend.

## Getting started

Here is a simple regression example:

```python
from aorun.models import Model
from aorun.layers import Dense, Activation

model = Model()
model.add(Dense(10, input_dim=3))
model.add(Activation('relu'))
model.add(Dense(1))

model.fit(X_train, y_train, loss='mse', optimizer='adam')

y_pred = model.predict(X_test)
```

## Why Aorun?

[Aorun](https://en.wikipedia.org/wiki/Aorun) is a Dinosaur. Dinosaurs are cool.
