import numpy as np


def sigmoid(Z):
    return 1.0 / (1.0 + np.exp(-Z))


def cross_entropy(Y_hat, Y_data):
    # ensure that no log(0) is done
    delta = 1e-7
    Y_hat += delta
    Y_data = Y_data.astype(float)
    Y_hat += delta

    to_sum = Y_data * np.log(Y_hat) + (1 - Y_data) * np.log((1 - Y_hat)) + delta
    return -np.sum(to_sum)
