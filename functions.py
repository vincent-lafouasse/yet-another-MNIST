import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def cross_entropy(Y_hat, Y_data):
    # ensure that no log(0) is done
    delta = 1e-7

    to_sum = Y_data * np.log(Y_hat + delta) + (1 - Y_data) * np.log((1 - Y_hat + delta))
    return -np.sum(to_sum)


def argmax(X):
    pass
