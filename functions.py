import numpy as np


class Sigmoid:
    def __init__(self):
        pass

    @staticmethod
    def f(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def deriv(x):
        return Sigmoid.f(x) * (1 - Sigmoid.f(x))


def cross_entropy(Y_hat, Y_data):
    # ensure that no log(0) is done
    epsilon = 1e-7

    loss = -np.mean(np.sum(Y_data * np.log(Y_hat + epsilon), axis=1))

    return loss


def grad_cross_entropy(Y_hat, Y_data):
    ones_true_class = np.zeros_like(Y_hat)
    ones_true_class[np.arange(len(Y_hat)), Y_data] = 1

    softmax = np.exp(Y_hat) / np.exp(Y_hat).sum(axis=-1, keepdims=True)

    return (-ones_true_class + softmax) / Y_hat.shape[0]


def argmax(X):
    pass
