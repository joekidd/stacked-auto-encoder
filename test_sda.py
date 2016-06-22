#!/usr/bin/env python
import deeplearn.datasets as dt
import deeplearn.metrics as mt
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
    ) = dt.load_horse_racing()
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    weights_initializer = np.random.RandomState(123)
    sdA = StackedDenoisingAutoencoder(
            x, y, 20, 14, [500, 500, 500], 
            weights_initializer,
            RandomStreams(weights_initializer.randint(2 ** 30)),
            [0.1, 0.2, 0.3]
    )
    epochs = 20
    batch_size = 20
    no_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    no_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size

    train_fns = sdA.get_pretrain_fns(index, batch_size, 0.04, train_x)
  
    pretrain = True
    if pretrain:  
        for train_fn in train_fns:
            for epoch in range(epochs):
                train_costs = []
                for batch_no in range(no_train_batches):
                    train_costs.append(train_fn(batch_no))
                print "Epoch %d: train_cost=%f" % (epoch, np.mean(train_costs))

    print "============== TRAINING ============="
    epochs = 100
    valid_fn = sdA.get_valid_fn(index, batch_size, train_x, train_y)
    alpha = 2.0
    for epoch in range(epochs):
        if epoch % 25 == 0:
            alpha /= 2
        train_fn = sdA.get_train_fn(
                index, batch_size, alpha, train_x, train_y
        )
        train_costs = []
        for batch_no in range(no_train_batches):
            train_costs.append(train_fn(batch_no))
        valid_costs = []
        for batch_no in range(no_valid_batches):
            valid_costs.append(valid_fn(batch_no))
        print "Epoch %d: train_cost=%f cv_cost=%f" % (epoch, np.mean(train_costs), np.mean(valid_costs))

    print "============= CONFUSION MATRIX ==========="
    given_y = sdA.get_predict_fn()(valid_x.get_value())
    expected_y = valid_y.eval()
    print mt.confusion_matrix(given_y, expected_y, list(set(expected_y)))
if __name__ == "__main__":
    main()
    
