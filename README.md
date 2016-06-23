# Stacked Denoising Autoencoders
Implementation of the SDA using theano library.

### How to run
1) ./setup.py
2) ./test_sda.py

### Directory description
data/training.csv - the horse racing dataset for training
data/testing.csv - the horse racing dataset for testing
setupy.py - a script preprocessing the data/testing and data/training into a
format to be used for machine learning
test_logistic_regression.py - a script for testing logistic regression
test_denoising_autoencoder.py - a script for testing denoising autoencoder
test_sda.py - a script for testing stacked denoising autoencoder
deeplearn/classifers.py - classifiers implementation
deeplearn/metrics.py - confusion matrix
deeplear/datasets.py - utils for generating datasets
