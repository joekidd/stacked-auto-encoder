#!/usr/bin/env python
import deeplearn.datasets as dt
from deeplearn.classifiers import LogisticRegression
import deeplearn.metrics as mt
import theano.tensor as T
import numpy as np

def main():
    (
        (train_x, train_y),
        (valid_x, valid_y),
        (test_x, test_y)
    ) = dt.load_iris()

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(x, y, 4, 3)

    epochs = 100
    batch_size = 10
    no_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    no_cv_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
    no_test_batches = test_x.get_value(borrow=True).shape[0] // batch_size

    train_fn = classifier.get_train_fn(index, batch_size, 0.04, train_x, train_y)
    valid_fn = classifier.get_valid_fn(index, batch_size, valid_x, valid_y)
    test_fn = classifier.get_valid_fn(index, batch_size, test_x, test_y)

    for epoch in range(epochs):
        train_costs = []
        for batch_no in range(no_train_batches):
            train_costs.append(train_fn(batch_no))
        cv_costs = []
        for batch_no in range(no_cv_batches):
            cv_costs.append(valid_fn(batch_no))
        print "Epoch %d: train_cost=%f cv_cost=%f" % (epoch, np.mean(train_costs), np.mean(cv_costs))

    test_cost = []
    for batch_no in range(no_test_batches):
        test_cost.append(test_fn(batch_no))
    print "Test set cost = %f" % (np.mean(test_cost))
    given_y = classifier.get_predict_fn()(test_x.get_value())
    expected_y = test_y.eval()
    print "\n=======> Confusion matrix:"
    print mt.confusion_matrix(given_y, expected_y, list(set(expected_y)))

if __name__ == "__main__":
    main()
