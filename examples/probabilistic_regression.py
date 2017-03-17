import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
from aorun.models import Model
from aorun.layers import ProbabilisticDense
from aorun.layers import Activation
from aorun.optimizers import SGD

X, y = datasets.load_boston(return_X_y=True)
X = X.astype('float32')
y = y.astype('float32')
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

model = Model()
model.add(ProbabilisticDense(3, input_dim=X_train.shape[-1]))
model.add(Activation('relu'))
model.add(ProbabilisticDense(2))
model.add(Activation('relu'))
model.add(ProbabilisticDense(1))

sgd = SGD(lr=0.005)
history = model.fit(X_train, y_train, loss='mse', optimizer=sgd, n_epochs=500)
y_pred = model.forward(X_test)
print('r2_score:', metrics.r2_score(y_test, y_pred))
print('mean_absolute_error:', metrics.mean_absolute_error(y_test, y_pred))
print('mean_squared_error:', metrics.mean_squared_error(y_test, y_pred))
