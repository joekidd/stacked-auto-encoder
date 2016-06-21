#!/usr/bin/env python
import deeplearn.datasets as dt
from deeplearn.classifiers import DenoisingAutoencoder
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

def main():
    (
        (train_x, train_y),
        (valid_x, valid_y),
        (test_x, test_y)
    ) = dt.load_iris()

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    weights_initializer = np.random.RandomState(123)
    dA = DenoisingAutoencoder(
            weights_initializer,
            RandomStreams(weights_initializer.randint(2 ** 30)),
            4, 500, 0.3, x
    )

    epochs = 15
    batch_size = 20
    no_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size

    train_fn = dA.get_train_fn(index, batch_size, 0.1, train_x)

    for epoch in range(epochs):
        train_costs = []
        for batch_no in range(no_train_batches):
            train_costs.append(train_fn(batch_no))
        print "Epoch %d: train_cost=%f" % (epoch, np.mean(train_costs))

if __name__ == "__main__":
    main()
    
