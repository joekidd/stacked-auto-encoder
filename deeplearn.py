import theano
import theano.tensor as T
import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle

class LogisticRegression(object):
    def __init__(self, theano_input, theano_output, no_in, no_out):
        self.theano_input = theano_input
        self.theano_output = theano_output
        self.weights = theano.shared(
            value=np.zeros(
                (no_in, no_out),
                dtype=theano.config.floatX
            ),
            name='weights',
            borrow=True
        )
        self.biases = theano.shared(
            value=np.zeros(
                (no_out,),
                dtype=theano.config.floatX
            ),
            name='biases',
            borrow=True
        )
        # the probability of y class given features x
        self.prob_y_given_x = T.nnet.softmax(
            T.dot(self.theano_input, self.weights) + self.biases
        )
        # the prediction is the max value of prob_y_given_x
        self.y_pred = T.argmax(self.prob_y_given_x, axis=1)
        self.params = [self.weights, self.biases]

    def cost(self):
        return -T.mean(
            T.log(self.prob_y_given_x)[T.arange(
                self.theano_output.shape[0]), self.theano_output]
            )

    def errors(self):
        return T.mean(T.neq(self.y_pred, self.theano_output))

    def get_train_fn(self, index, batch_size, learning_rate, train_set_x, train_set_y):
        grad_weights = T.grad(cost=self.cost(), wrt=self.weights)
        grad_biases = T.grad(cost=self.cost(), wrt=self.biases)
        updates = [(self.weights, self.weights - learning_rate * grad_weights),
                   (self.biases, self.biases - learning_rate * grad_biases)]
        train_fn = theano.function(
            inputs=[index],
            outputs=self.cost(),
            updates=updates,
            givens={
                self.theano_input : train_set_x[index * batch_size: (index + 1) * batch_size],
                self.theano_output : train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        return train_fn

    def get_valid_fn(self, index, batch_size, train_set_x, train_set_y):
        valid_fn = theano.function(
            inputs=[index],
            outputs=self.errors(),
            givens={
                self.theano_input : train_set_x[index * batch_size: (index + 1) * batch_size],
                self.theano_output : train_set_y[index * batch_size: (index + 1) * batch_size]
            }
        )
        return valid_fn

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

def generate_iris_dataset():
    iris = datasets.load_iris()
    x, y = shuffle(iris.data, iris.target)
    size = iris.target.shape[0]
    train_size = int(0.6 * size)
    cv_size = int(0.8 * size)
    return (
        make_shared(x[:train_size], y[:train_size]),
        make_shared(x[train_size:cv_size], y[train_size:cv_size]),
        make_shared(x[cv_size:], y[cv_size:]))

def test_logistic_regression():
    (
        (train_x, train_y),
        (valid_x, valid_y),
        (test_x, test_y)
    ) = generate_iris_dataset()

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(x, y, 4, 3)

    epochs = 50
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
        print "Epoch %d train_cost=%f cv_cost=%f" % (epoch, np.mean(train_costs), np.mean(cv_costs))

    test_cost = []
    for batch_no in range(no_test_batches):
        test_cost.append(test_fn(batch_no))
    print "Test set cost = %f" % (np.mean(test_cost))

test_logistic_regression()
