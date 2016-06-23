import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

import numpy as np

class LogisticRegression(object):
    """Implementation of the logistic layer"""
    def __init__(self, theano_input, theano_output, no_in, no_out):
        self.theano_input = theano_input
        self.theano_output = theano_output
        self.weights = theano.shared(
            value=np.zeros(
                (no_in, no_out),
                dtype=theano.config.floatX
            ),
            borrow=True
        )
        self.biases = theano.shared(
            value=np.zeros(
                (no_out,),
                dtype=theano.config.floatX
            ),
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
        """Returns the mean of negative log likelihood"""
        return -T.mean(
            T.log(self.prob_y_given_x)[T.arange(
                self.theano_output.shape[0]), self.theano_output]
            )

    def errors(self):
        """The ratio of errors in the minibatch"""
        return T.mean(T.neq(self.y_pred, self.theano_output))

    def get_train_fn(
            self, index, batch_size, learning_rate, train_set_x, train_set_y
    ):
        """Returns symbolic training function"""
        grad_weights = T.grad(cost=self.cost(), wrt=self.weights)
        grad_biases = T.grad(cost=self.cost(), wrt=self.biases)
        updates = [(self.weights, self.weights - learning_rate * grad_weights),
                   (self.biases, self.biases - learning_rate * grad_biases)]
        train_fn = theano.function(
            inputs=[index],
            outputs=self.cost(),
            updates=updates,
            givens={
                self.theano_input : train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.theano_output : train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        return train_fn

    def get_valid_fn(self, index, batch_size, cv_set_x, cv_set_y):
        """Returns symbolic function to be used for cross validation"""
        valid_fn = theano.function(
            inputs=[index],
            outputs=self.errors(),
            givens={
                self.theano_input : cv_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.theano_output : cv_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        return valid_fn

    def get_predict_fn(self):
        """Returns symbolic function to be used for prediction"""
        predict_fn = theano.function(
            inputs=[self.theano_input],
            outputs=self.y_pred
        )
        return predict_fn

class HiddenLayer(object):
    """Implementation of a single layer of MLP"""
    def __init__(
            self,
            weights_initializer,
            theano_input,
            no_in,
            no_out,
            weights=None,
            biases=None,
            activation=None
    ):
        self.theano_input = theano_input
        if weights is None:
            tmp = np.asarray(
                weights_initializer.uniform(
                    low=-np.sqrt(6. / (no_in + no_out)),
                    high=np.sqrt(6. / (no_in + no_out)),
                    size=(no_in, no_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == T.nnet.sigmoid:
                tmp = tmp * 4
            weights = theano.shared(value=tmp, borrow=True)

        if biases is None:
            biases = np.zeros((no_out,), dtype=theano.config.floatX)
            biases = theano.shared(value=biases, borrow='True')

        self.weights = weights
        self.biases = biases

        self.output = activation(
            T.dot(self.theano_input, self.weights) + self.biases
        )
        self.params = [self.weights, self.biases]

class DenoisingAutoencoder(object):
    """Implementation of a denosing autoencoder"""
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
        """The cost functions tries to measure how close is the reconstructed
        input to the original one
        """
        # corrupt the original input
        corrupted_input = self.noise_initializer.binomial(
            size=self.theano_input.shape,
            n=1,
            p=1 - self.corruption_level,
            dtype=theano.config.floatX
        ) * self.theano_input
        # pass it through hidden layer (with shared weights)
        hidden_values = T.nnet.sigmoid(
            T.dot(corrupted_input, self.weights) + self.biases
        )
        # reconstruct the input
        reconstructed_input = T.nnet.sigmoid(
            T.dot(hidden_values, self.prime_weights) + self.prime_biases
        )
        # return the 'similiarity'
        return T.mean(
            -T.sum(
                self.theano_input * T.log(reconstructed_input)
                + (1 - self.theano_input) * T.log(1 - reconstructed_input),
                axis=1
            )
        )

    def get_train_fn(self, index, batch_size, learning_rate, train_set, in_var=None):
        """Returns symbolic function to be used for training"""
        if in_var is None:
            in_var = self.theano_input
        cost = self.cost()
        grad_params = T.grad(cost, self.params)
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, grad_params)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                in_var : train_set[index * batch_size: (index + 1) * batch_size]
            },
        )
        return train_fn

class StackedDenoisingAutoencoder(object):
    """Implementation of the stacked denoising autoencoder"""
    def __init__(
            self,
            theano_input,
            theano_output,
            no_ins,
            no_outs,
            layers,
            weight_initializer,
            noise_initializer,
            corruption_levels
    ):
        self.theano_input = theano_input
        self.theano_output = theano_output
        self.no_layers = len(layers)
        self.sigmoid_layers = []
        self.da_layers = []
        self.params = []

        for layer in range(self.no_layers):
            if layer == 0:
                input_size = no_ins
                layer_input = self.theano_input
            else:
                input_size = layers[layer - 1]
                layer_input = self.sigmoid_layers[layer - 1].output

            sigmoid_layer = HiddenLayer(
                weight_initializer,
                layer_input,
                input_size,
                layers[layer],
                activation=T.nnet.sigmoid
            )
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)

            da_layer = DenoisingAutoencoder(
                weight_initializer,
                noise_initializer,
                input_size,
                layers[layer],
                corruption_levels[layer],
                layer_input,
                weights=sigmoid_layer.weights,
                bhid=sigmoid_layer.biases
            )
            self.da_layers.append(da_layer)

        self.log_layer = LogisticRegression(
            self.sigmoid_layers[-1].output,
            self.theano_output,
            layers[-1],
            no_outs
        )
        self.params.extend(self.log_layer.params)

    def get_pretrain_fns(self, index, batch_size, learning_rate, train_set):
        """Return symbolic function to be used for the denoising autoencoder
        weights intialization.
        """
        pretrain_fns = []
        for dA in self.da_layers:
            pretrain_fns.append(
                dA.get_train_fn(index, batch_size, learning_rate, train_set, self.theano_input)
            )
        return pretrain_fns

    def get_train_fn(
            self, index, batch_size, learning_rate, train_set_x, train_set_y
    ):
        """Returns symbolic function to be used for MLP training """
        train_cost = self.log_layer.cost()
        grad_params = T.grad(train_cost, self.params)
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, grad_params)
        ]
        return theano.function(
            inputs=[index],
            outputs=train_cost,
            updates=updates,
            givens={
                self.theano_input : train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.theano_output : train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

    def get_valid_fn(self, index, batch_size, cv_set_x, cv_set_y):
        """Returns symbolic function to be used for crossvalidation """
        return theano.function(
            inputs=[index],
            outputs=self.log_layer.errors(),
            givens={
                self.theano_input : cv_set_x[
                    index * batch_size : (index + 1) * batch_size
                ],
                self.theano_output : cv_set_y[
                    index * batch_size : (index + 1) * batch_size
                ]
            }
        )

    def get_predict_fn(self):
        """Returns symbolic function to be used for prediction """
        return theano.function(
            inputs=[self.theano_input],
            outputs=self.log_layer.y_pred
        )
