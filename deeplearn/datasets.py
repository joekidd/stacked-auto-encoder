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

def split_dataset(x, y):
    x, y = shuffle(x, y)
    size = y.shape[0]
    train_size = int(0.6 * size)
    cv_size = int(0.8 * size)
    return (
        make_shared(x[:train_size], y[:train_size]),
        make_shared(x[train_size:cv_size], y[train_size:cv_size]),
        make_shared(x[cv_size:], y[cv_size:]))

def load_iris():
    iris = datasets.load_iris()
    return split_dataset(iris.data, iris.target)

def load_random(samples, features, classes):
    (x, y) = datasets.make_classification(
        n_samples=samples,
        n_features=features,
        n_informative=features/2,
        n_classes=classes,
        shuffle=True
    )
    return split_dataset(x, y)

def load_horse_racing():
    train_x = np.genfromtxt("data/train.csv", delimiter=',')
    test_x = np.genfromtxt("data/test.csv", delimiter=',')
    train_y = np.genfromtxt("data/train.labels.csv", delimiter=',')
    test_y = np.genfromtxt("data/test.labels.csv", delimiter=',')
    train_x, train_y = shuffle(train_x, train_y)
    test_x, test_y = shuffle(test_x, test_y)
    half = test_y.shape[0] // 2
    return (
        make_shared(train_x, train_y),
        make_shared(test_x[:half], test_y[:half]),
        make_shared(test_x[half:], test_y[half:])
    )
