import numpy as np

from Dataset import Dataset, TrainingSet, TestingSet


def softmax():
    pass


def cross_entropy():
    pass


class Layer:
    def __init__(self):
        self.W = np.random.rand(784, 10)
        self.b = np.random.rand(1, 10)

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
    print(training_data.X[0])
    print(training_data.Y[0])


if __name__ == "__main__":
    main()
