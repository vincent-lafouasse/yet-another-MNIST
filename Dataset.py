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
        pass


class TrainingSet(Dataset):
    def __init__(self):
        super().__init__(TRAIN_PATH)


class TestingSet(Dataset):
    def __init__(self):
        super().__init__(TEST_PATH)
