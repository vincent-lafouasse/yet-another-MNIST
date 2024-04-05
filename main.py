import numpy as np

from Dataset import Dataset, TrainingSet, TestingSet
from functions import softmax, cross_entropy


class FullyConnectedLayer:
    def __init__(self, n_features, n_classes):
        self.W = np.random.rand(n_features, n_classes)
        self.b = np.random.rand(1, n_classes)

    def forward(self, X):
        self.Z = np.dot(X, self.W) + self.b
        self.Y_hat = softmax(self.Z)
        return self.Y_hat

    def backward(self, Y_data):
        dZ = self.Y_hat - Y_data
        q = 10
        db = (1 / q) * dZ


def main():
    training_data = TrainingSet()

    layer = FullyConnectedLayer(784, 10)
    print(layer)


if __name__ == "__main__":
    main()
