from torchvision import datasets, transforms
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class Data:
    train_set: Tuple[np.ndarray, np.ndarray]
    test_set: Tuple[np.ndarray, np.ndarray]


def get_data() -> Data:
    train_data = datasets.MNIST(
        root="./.data", train=True, download=True, transform=transforms.ToTensor()
    )
    test_data = datasets.MNIST(
        root="./.data", train=False, download=True, transform=transforms.ToTensor()
    )

    X_train = train_data.data.numpy()
    y_train = train_data.targets.numpy()
    X_train = X_train.reshape((X_train.shape[0], -1)) / 255

    X_test = test_data.data.numpy()
    y_test = test_data.targets.numpy()
    X_test = X_test.reshape((X_test.shape[0], -1)) / 255

    return Data(train_set=(X_train, y_train), test_set=(X_test, y_test))
