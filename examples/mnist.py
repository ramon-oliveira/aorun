import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics

import numpy as np
from aorun import datasets
from aorun.models import Model
from aorun.layers import Conv2D
from aorun.layers import Dense
from aorun.layers import Flatten
from aorun.layers import Activation
from aorun.layers import Dropout

(X, y), (X_test, y_test) = datasets.load_mnist()
X = X / 127.0
X_test = X_test / 127.0
y = np.eye(y.max() + 1)[y]
y_test = np.eye(y_test.max() + 1)[y_test]
print(X.shape, X_test.shape)

X = X.astype('float32')
X_test = X_test.astype('float32')
y = y.astype('float32')
y_test = y_test.astype('float32')

model = Model(
    Conv2D(8, kernel_size=(3, 3), input_dim=X.shape[1:]),
    Flatten(),
    Activation('relu'),
    Dropout(0.5),
    Dense(100),
    Activation('relu'),
    Dropout(0.5),
    Dense(y_test.shape[-1]),
    Activation('softmax')
)

loss = 'categorical_crossentropy'
history = model.fit(X, y, loss=loss, val_data=(X_test, y_test))

y_pred = model.predict(X_test)
acc = metrics.accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
print('Classes:', y.shape[1])
print('Accuracy:', acc)
