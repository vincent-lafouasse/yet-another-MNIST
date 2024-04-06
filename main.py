import numpy as np

from Dataset import Dataset, TrainingSet, TestingSet, XORDataset
from functions import Sigmoid, ReLU, cross_entropy, grad_cross_entropy


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

    def backward(self, Y, grad_output, learning_rate):
        self.grad_input = grad_output @ self.W.T

        self.grad_weights = value.T @ grad_output
        self.grad_bias = np.sum(grad_output, axis=0, keepdims=True)

        self.W -= self.learning_rate * self.grad_weights
        self.b -= self.learning_rate * self.grad_bias
        return self.grad_input


class ReLULayer(Layer):
    def __init__(self):
        pass

    def forward(self, X):
        return ReLU.f(X)

    def backward(self, value, grad):
        return grad * ReLU.deriv(value)


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

    layers = [
        FullyConnectedLayer(n_features, 16),
        FullyConnectedLayer(16, n_classes),
        ReLULayer(),
    ]

    layer_inputs = [data.X]
    input = data.X

    for layer in layers:
        input = layer.forward(input)
        layer_inputs.append(input)

    prediction = layer_inputs.pop()

    print("layer inputs:")
    print(len(layer_inputs))
    print("prediction")
    print(prediction)

    loss = cross_entropy(prediction, data.Y)
    print(f"Loss: {loss}")
    # loss_grad = grad_cross_entropy(prediction, data.Y)

    # print(loss_grad)


if __name__ == "__main__":
    main()
