import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import functools
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import numpy as np
from aorun.models import Model
from aorun.layers import ProbabilisticDense
from aorun.layers import Activation
from aorun.optimizers import SGD
from aorun.losses import variational_loss

X, y = datasets.load_digits(return_X_y=True)
X = X.astype('float32')
y = np.eye(y.max() + 1)[y].astype('float32')
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)

model = Model(
    ProbabilisticDense(200, input_dim=X_train.shape[-1]),
    Activation('relu'),
    ProbabilisticDense(200),
    Activation('relu'),
    ProbabilisticDense(y_test.shape[-1]),
    Activation('softmax')
)

opt = SGD(lr=0.01, momentum=0.99)
loss = variational_loss(model, 'categorical_crossentropy')
history = model.fit(X_train, y_train, epochs=20, loss=loss, optimizer=opt)

y_pred = model.forward(X_test)
acc = metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print('test samples:', len(y_test))
print('classes:', len(y_test[0]))
print('Accuracy:', acc)
