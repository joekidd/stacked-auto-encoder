#!/usr/bin/env python
import deeplearn.datasets as dt
from deeplearn.classifiers import StackedDenoisingAutoencoder
import theano.tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
import theano

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
    sdA = StackedDenoisingAutoencoder(
            x, y, 4, 3, [500, 600, 500], 
            weights_initializer,
            RandomStreams(weights_initializer.randint(2 ** 30)),
            [0.3, 0.3, 0.4]
    )

    epochs = 50
    batch_size = 20
    no_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size

    train_fns = sdA.get_pretrain_fns(index, batch_size, 0.1, train_x)
    
    for train_fn in train_fns:
        for epoch in range(epochs):
            train_costs = []
            for batch_no in range(no_train_batches):
                train_costs.append(train_fn(batch_no))
            print "Epoch %d: train_cost=%f" % (epoch, np.mean(train_costs))

if __name__ == "__main__":
    main()
    
