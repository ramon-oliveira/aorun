from .context import aorun

from aorun.models import Model
from aorun.layers import Dense, ProbabilisticDense, Relu


def test_model_constructor_empty():

    model = Model()

    assert len(model.layers) == 0


def test_model_constructor_layers():

    model = Model(
        Dense(10),
        Relu(),
        Dense(1),
        ProbabilisticDense(20)
    )

    assert len(model.layers) == 4
    assert type(model.layers[0]) == Dense
    assert type(model.layers[3]) == ProbabilisticDense


def test_model_add_layers():

    model = Model()
    model.add(Dense(10))
    model.add(Relu())
    model.add(Dense(1))
    model.add(ProbabilisticDense(20))

    assert len(model.layers) == 4
    assert type(model.layers[0]) == Dense
    assert type(model.layers[3]) == ProbabilisticDense
