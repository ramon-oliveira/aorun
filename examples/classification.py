import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import torch
import numpy as np
from aorun.models import Model
from aorun.layers import Dense
from aorun.layers import Activation
from aorun.optimizers import SGD

torch.manual_seed(42)

X, y = datasets.load_iris(return_X_y=True)
X = X.astype('float32')
y = np.eye(y.max()+1)[y].astype('float32')

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)

X_train = torch.from_numpy(X_train)
y_train = torch.from_numpy(y_train)
X_test = torch.from_numpy(X_test)
y_test = torch.from_numpy(y_test)

model = Model(
    Dense(10, input_dim=X_train.size()[-1]),
    Activation('relu'),
    Dense(10),
    Activation('relu'),
    Dense(y_test.size()[-1]),
    Activation('softmax')
)

sgd = SGD(lr=0.5)
history = model.fit(X_train, y_train, n_epochs=100,
                    loss='categorical_crossentropy', optimizer=sgd)

y_test = y_test.numpy()
y_pred = model.forward(X_test).data.numpy()
acc = metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print('Accuracy:', acc)
