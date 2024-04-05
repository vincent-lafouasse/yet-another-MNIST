import numpy as np


def softmax():
    pass


def cross_entropy():
    pass

DATASET_PATH = {'X' : 0, "Y": 0}

class Dataset:
    def __init__(self, path):
        self.X = idx2numpy.convert_from_file(path["X"])
        self.Y = idx2numpy.convert_from_file(path["Y"])


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
    print("hello")


if __name__ == "__main__":
    main()
