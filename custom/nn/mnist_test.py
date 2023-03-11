import numpy as np

from neural_network import *
from sklearn.preprocessing import OneHotEncoder

DATA_PATH = "archive"


def get_data():
    train_data = np.loadtxt(DATA_PATH + "/mnist_train.csv", skiprows=1, max_rows=1000, delimiter=',')
    test_data = np.loadtxt(DATA_PATH + "/mnist_test.csv", skiprows=1,max_rows=1000, delimiter=',')
    y_train = train_data[:, :1].reshape(-1, 1).astype("int32")
    x_train = train_data[:, 1:].reshape(-1, 28, 28, 1).astype("float32")
    y_test = test_data[:, 0].reshape(-1, 1).astype("int32")
    x_test = test_data[:, 1:].reshape(-1, 28, 28, 1).astype("float32")
    return x_train, x_test, y_train, y_test


class MnistModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = [
            Convolution(3, 1, 10), Relu(),
            Convolution(3, 10, 64), Relu(),
            Flatten(),
            Dense(36864, 10), SoftMax()
        ]

    def _forward_raw_data(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def optimize(self, gradients, lr):
        for layer in self.layers[::-1]:
            gradients = layer.backward(gradients, lr)


def main():
    x_train, x_test, y_train, y_test = get_data()
    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y_train)
    model = MnistModel()
    model.fit(x_train, y, cat_cross_entropy, 0.01, batch_size=32, max_epochs=20)
    y_pred = model(x_test)
    acc = (encoder.inverse_transform(y_pred) == y_test).sum() / len(y_pred)
    print(f"Acc: {acc}")


if __name__ == "__main__":
    main()
