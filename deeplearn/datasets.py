import theano
import theano.tensor as T
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle


def make_shared(x, y):
    shared_x = theano.shared(
        np.asarray(x, dtype=theano.config.floatX),
        borrow=True
    )
    shared_y = theano.shared(
        np.asarray(y, dtype=theano.config.floatX),
        borrow=True
    )
    return shared_x, T.cast(shared_y, 'int32')

def load_iris():
    iris = datasets.load_iris()
    x, y = shuffle(iris.data, iris.target)
    size = iris.target.shape[0]
    train_size = int(0.6 * size)
    cv_size = int(0.8 * size)
    return (
        make_shared(x[:train_size], y[:train_size]),
        make_shared(x[train_size:cv_size], y[train_size:cv_size]),
        make_shared(x[cv_size:], y[cv_size:]))


