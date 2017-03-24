from .context import aorun

from aorun import datasets


def test_mnist():
    (X_train, y_train), (X_test, y_test) = datasets.load_mnist()

    assert len(X_train) == 60000
    assert len(y_train) == 60000
    assert len(X_test) == 10000
    assert len(y_test) == 10000
