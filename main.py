import numpy as np

from Dataset import Dataset, TrainingSet, TestingSet, XORDataset
from functions import Sigmoid, ReLU, Softmax, cross_entropy, grad_cross_entropy


class Layer:
    def __init(self, n_features, n_classes):
        raise NotImplementedError("base class methods are virtual")

    def forward(self, X):
        raise NotImplementedError("base class methods are virtual")

    def backward(self, Y_data):
        raise NotImplementedError("base class methods are virtual")


class FullyConnectedLayer(Layer):
    def __init__(self, n_features, n_classes):
        self.W = np.random.rand(n_features, n_classes)
        self.b = np.random.rand(1, n_classes)

    def forward(self, X):
        self.Z = X @ self.W + self.b
        return self.Z

    # Y = target output?
    # dC/dw_i = (dC/da) * (da/dz) * (dz/dw_i)
    # same but replace w with b
    def backward(self, Y, grad_output, learning_rate):
        self.grad_input = grad_output @ self.W.T

        self.grad_weights = value.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        self.W -= self.learning_rate * self.grad_weights
        self.b -= self.learning_rate * self.grad_bias
        return self.grad_input


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)


def main():
    data = XORDataset()
    n_features = len(data.X[0])
    n_classes = len(data.Y[0])

    layers = [
        FullyConnectedLayer(n_features, 16),
        FullyConnectedLayer(16, n_classes),
    ]

    activation = data.X
    activations = [data.X]
    zs = []

    # forward
    for layer in layers:
        z = layer.forward(activation)
        zs.append(z)
        activation = Sigmoid.f(z)
        activations.append(activation)

    print(zs)
    print(activations)

    # backward
    for layer in layers:
        print(layer.b.shape)
    grad_w = [np.zeros(layer.W.shape) for layer in layers]
    grad_b = [np.zeros(layer.b.shape) for layer in layers]
    print(grad_b)


if __name__ == "__main__":
    main()
