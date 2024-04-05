import numpy as np

from Dataset import Dataset, TrainingSet, TestingSet, XORDataset
from functions import sigmoid, cross_entropy


class Layer:
    def __init(self, n_features, n_classes):
        pass

    def forward(self, X):
        pass

    def backward(self, Y_data):
        pass


class FullyConnectedLayer(Layer):
    def __init__(self, n_features, n_classes):
        self.W = np.random.rand(n_features, n_classes)
        self.b = np.random.rand(1, n_classes)

    def forward(self, X):
        self.Z = np.dot(X, self.W) + self.b
        self.Y_hat = sigmoid(self.Z)
        return self.Y_hat

    def backward(self, Y_data):
        dZ = self.Y_hat - Y_data
        q = 10
        db = (1 / q) * dZ


class ActivationLayer(Layer):
    def __init__(self):
        pass


class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)


def main():
    data = XORDataset()
    print(data.X)
    print(data.Y)
    n_features = 2
    n_classes = 2

    layer = FullyConnectedLayer(n_features, n_classes)
    Y_hat = layer.forward(data.X)
    print(Y_hat)

    loss = cross_entropy(Y_hat, data.Y)
    print(f"loss = {loss}")


if __name__ == "__main__":
    main()
