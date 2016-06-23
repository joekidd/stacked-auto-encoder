#!/usr/bin/env python
from __future__ import print_function

import numpy as np

import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import deeplearn.datasets as dt
import deeplearn.metrics as mt
from deeplearn.classifiers import StackedDenoisingAutoencoder

def main():
    pretrain_epochs = 15
    train_epochs = 50
    batch_size = 20
    enable_pretraining = True
    #learning rate
    pretrain_lr = 0.1
    lr = 2.0
    decrease_lr_freq = 5
    decrease_lr = 2.0
    weights_initializer = np.random.RandomState(321)

    # load the training set
    (
        (train_x, train_y),
        (valid_x, valid_y),
        (test_x, test_y)
    ) = dt.load_horse_racing()


    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    sdA = StackedDenoisingAutoencoder(
        x, y, 20, 14, [500, 500, 500],
        weights_initializer,
        RandomStreams(weights_initializer.randint(2 ** 30)),
        [0.1, 0.2, 0.3]
    )
    no_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    no_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size

    print("============ PRETRAINING ===========")
    if enable_pretraining:
        train_fns = sdA.get_pretrain_fns(
            index, batch_size, pretrain_lr, train_x
        )
        for layer, train_fn in enumerate(train_fns):
            for epoch in range(pretrain_epochs):
                train_costs = []
                for batch_no in range(no_train_batches):
                    train_costs.append(train_fn(batch_no))
                print("Layer=%d, epoch=%d: cost=%f"
                      % (layer, epoch + 1, np.mean(train_costs))
                     )

    print("============== TRAINING =============")
    valid_fn = sdA.get_valid_fn(index, batch_size, valid_x, valid_y)
    for epoch in range(train_epochs):
        # decrease the learning rate with time
        if (epoch + 1) % decrease_lr_freq == 0:
            lr /= decrease_lr
            print("Modified larning rate: %f" % lr)
        # get train function (with new lr)
        train_fn = sdA.get_train_fn(
            index, batch_size, lr, train_x, train_y
        )
        train_costs = []
        valid_costs = []
        for batch_no in range(no_train_batches):
            train_costs.append(train_fn(batch_no))
        for batch_no in range(no_valid_batches):
            valid_costs.append(valid_fn(batch_no))
        print("Epoch=%d cost=%f cv_error=%f%%"
              % (epoch + 1, np.mean(train_costs), np.mean(valid_costs) * 100)
             )

    print("============= CONFUSION MATRIX ===========")
    given_y = sdA.get_predict_fn()(test_x.get_value())
    expected_y = test_y.eval()
    print(mt.confusion_matrix(given_y, expected_y, list(set(expected_y))))

if __name__ == "__main__":
    main()
