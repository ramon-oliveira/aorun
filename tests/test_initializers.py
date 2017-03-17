import pytest
from .context import aorun

import math
from torch import Tensor
from aorun.layers import Dense
from aorun import initializers


def test_initializer_glorot_uniform():
    X = Tensor([[10, 10, 10]])
    input_dim = X.size()[-1]
    dense = Dense(10, init='glorot_uniform', input_dim=input_dim)

    assert dense.W.max() <= math.sqrt(6 / (input_dim + 10))
    assert dense.W.min() >= -math.sqrt(6 / (input_dim + 10))


def test_initializer_get():
    with pytest.raises(Exception) as e:
        initializers.get('UNKNOWN_TEST')
    with pytest.raises(Exception) as e:
        initializers.get(123)

    def init_test(w, a, b):
        return w
    assert initializers.get(init_test) == init_test
