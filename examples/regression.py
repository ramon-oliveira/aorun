import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
from aorun.models import Model
from aorun.layers import Dense
from aorun.layers import Activation
from aorun.optimizers import SGD

X, y = load_boston(return_X_y=True)
X = X.astype('float32')
y = y.astype('float32')

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

model = Model(
    Dense(10, input_dim=X_train.size()[-1]),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(1)
)

sgd = SGD(lr=0.001)
history = model.fit(X_train, y_train, loss='mse', optimizer=sgd, n_epochs=100)
y_pred = model.forward(X_test).data
r2 = metrics.r2_score(y_test.numpy(), y_pred.numpy())
print('r2 score:', r2)
