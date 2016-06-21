import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np

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

class HiddenLayer(object):
    def __init__(
            self,
            random_state,
            theano_input,
            theano_output,
            no_in,
            no_out,
            weights=None,
            biases=None,
            activation=None
    ):
        self.theano_input = theano_input
        self.theano_output = theano_output
        
        if weights is None:
            weights = np.asarray(
                random_state.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            weights = theano.shared(
                value=weights, name='weights', borrow=True)

        if biases is None:
            biases = numpy.zeros((n_out,), dtype=theano.config.floatX)
            biases = theano.shared(value=biases, name='biases', borrow='True')
        
        self.weights = weights
        self.biases = biases
        
        if activation is None:
            activation = T.tanh
        self.output = activation(
            T.dot(self.theano_input, self.weights) + self.biases)
        self.params = [self.weights, self.biases]

class DenoisingAutoencoder(object):
    def __init__(
        self,
        weights_initializer,
        noise_initializer,
        no_visible,
        no_hidden,
        corruption_level=0.0,
        theano_input=None,
        weights=None,
        bhid=None,
        bvis=None
    ):
        self.no_visible = no_visible
        self.no_hidden = no_hidden
        self.corruption_level = corruption_level 
        self.theano_input = theano_input
        self.noise_initializer = noise_initializer

        if weights is None:
            weights = np.asarray(
                weights_initializer.uniform(
                    low=-4 * np.sqrt(6. / (no_visible + no_hidden)),
                    high=4 * np.sqrt(6. / (no_visible + no_hidden)),
                    size=(no_visible, no_hidden)
                ),
                dtype=theano.config.floatX
            )
            weights = theano.shared(value=weights, name='weights', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=np.zeros(
                    no_visible, dtype=theano.config.floatX
                ),
                borrow=True
            )
        
        if not bhid:
            bhid = theano.shared(
                value=np.zeros(
                    no_hidden, dtype=theano.config.floatX
                ),
                borrow=True
            )
        
        self.weights = weights
        self.biases = bhid
        self.prime_biases = bvis
        self.prime_weights = self.weights.T
        self.params = [self.weights, self.biases, self.prime_biases]

    def cost(self):
        corrupted_input = self.noise_initializer.binomial(
            size=self.theano_input.shape,
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX
        ) * self.theano_input
        hidden_values = T.nnet.sigmoid(
            T.dot(corrupted_input, self.weights) + self.biases
        )
        reconstructed_input = T.nnet.sigmoid(
                T.dot(hidden_values, self.prime_weights) + self.prime_biases
        )
        return T.mean(
            -T.sum(
                self.theano_input * T.log(reconstructed_input)
                + (1 - self.theano_input) * T.log(1 - reconstructed_input),
                axis=1
            )
        )

    def get_train_fn(self, index, batch_size, learning_rate, train_set):
        grad_params = T.grad(self.cost(), self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, grad_params)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.cost(),
            updates=updates,
            givens={
                self.theano_input: train_set[index * batch_size: (index + 1) * batch_size]
            }
        )
        return train_fn
