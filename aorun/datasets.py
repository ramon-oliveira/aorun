import numpy as np
from . import utils


def load_mnist():
    url = 'https://s3.amazonaws.com/img-datasets/mnist.npz'
    filepath = utils.get_file(url, cache_subdir='datasets')
    d = np.load(filepath)
    return (d['x_train'], d['y_train']), (d['x_test'], d['y_test'])
