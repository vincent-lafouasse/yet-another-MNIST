import numpy as np
import idx2numpy


TRAIN_PATH = {"X": "data/train-images.idx3-ubyte", "Y": "data/train-labels.idx1-ubyte"}
TEST_PATH = {"X": "data/t10k-images.idx3-ubyte", "Y": "data/t10k-labels.idx1-ubyte"}


class Dataset:
    def __init__(self, path):
        self.X = idx2numpy.convert_from_file(path["X"])
        self.Y = idx2numpy.convert_from_file(path["Y"])
        self.process_data()

    def process_data(self):
        self.Y = one_hot(self.Y)
        self.X = flatten(self.X)
        self.X = self.X / 255  # pixel luminosity as a float in [0, 1] rather than u8


class TrainingSet(Dataset):
    def __init__(self):
        super().__init__(TRAIN_PATH)


class TestingSet(Dataset):
    def __init__(self):
        super().__init__(TEST_PATH)


def flatten(images):
    return np.array([np.concatenate(image) for image in images])


def one_hot_encode(label):
    out = np.zeros(10, dtype=int)
    out[label] = 1
    return out


def one_hot(labels):
    return [one_hot_encode(label) for label in labels]


class XORDataset:
    def __init__(self):
        self.X = np.array(
            [
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
                [1.0, 0.0],
            ]
        )
        self.Y = np.array(
            [
                [1, 0],
                [0, 1],
                [0, 1],
                [1, 0],
            ]
        )
