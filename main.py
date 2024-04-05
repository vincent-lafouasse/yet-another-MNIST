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
        self.A = sigmoid(self.Z)
        return self.A

    def backward(self, Y_data, learning_rate):
        pass


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
    n_features = len(data.X[0])
    n_classes = len(data.Y[0])

    layer1 = FullyConnectedLayer(n_features, 16)
    layer2 = FullyConnectedLayer(16, n_classes)

    hidden_activation = layer1.forward(data.X)
    prediction = layer2.forward(hidden_activation)
    print(prediction)

    loss = cross_entropy(prediction, data.Y)
    print(f"loss = {loss}")


if __name__ == "__main__":
    main()
