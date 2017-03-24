import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import numpy as np
from aorun.models import Model
from aorun.layers import Dense
from aorun.layers import Activation

X, y = datasets.load_digits(return_X_y=True)
X = X.astype('float32')
y = np.eye(y.max() + 1)[y].astype('float32')

X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape, y_train.shape)

model = Model(
    Dense(100, input_dim=X_train.shape[-1]),
    Activation('relu'),
    Dense(100),
    Activation('relu'),
    Dense(y_test.shape[-1]),
    Activation('softmax')
)

loss = 'categorical_crossentropy'
history = model.fit(X_train, y_train, loss=loss, val_split=0.1)

y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print('Classes:', y.shape[1])
print('Accuracy:', acc)
