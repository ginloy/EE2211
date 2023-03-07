from abc import abstractmethod

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def cat_cross_entropy(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = -np.sum(target * np.log(np.clip(pred, 1e-7, None))) / pred.shape[0]
    derivative = (pred - target) / pred.shape[0]
    return error, derivative


def mse(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = np.mean(np.sum((pred - target) ** 2, axis=1))
    gradient = (2 * (pred - target)) / len(pred)
    return error, gradient


def normalize(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


class Layer:
    @abstractmethod
    def forward(self, inp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self, inp_dim: int, out_dim: int):
        self.__input = None
        self.__weight = np.random.normal(loc=0, scale=np.sqrt(2 / inp_dim), size=(inp_dim, out_dim))
        self.__bias = np.zeros((1, out_dim))

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # print(self.__weight)
        self.__input = inp
        return inp @ self.__weight + self.__bias

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        mean_input = np.mean(self.__input, axis=0, keepdims=True)
        # print(output_gradient, mean_input)
        prev_weight = self.__weight[:, :]
        self.__weight -= learn_rate * self.__input.T @ output_gradient
        self.__bias -= learn_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return output_gradient @ prev_weight.T


class LeakyRelu(Layer):

    def __init__(self, alpha=0.01):
        self.__alpha = alpha
        self.__input = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        return np.maximum(inp, inp * self.__alpha)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        temp = np.ones(self.__input.shape)
        temp[self.__input < 0] = self.__alpha
        return temp * output_gradient
        # return learn_rate * (output_gradient > 0)


class Tanh(Layer):

    def __init__(self):
        self.__input = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        return np.tanh(inp)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        return (1 - np.tanh(self.__input) ** 2) * output_gradient


class SoftMax(Layer):

    def __init__(self):
        self.__output = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        prep = inp - np.max(inp, axis=1, keepdims=True)
        self.__output = np.exp(prep) / np.sum(np.exp(prep), axis=1, keepdims=True)
        return self.__output

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        batch_size = output_gradient.shape[0]
        classes = output_gradient.shape[1]
        jacob_matrix = np.zeros((batch_size, classes, classes))
        diag = np.arange(classes)
        jacob_matrix[:, diag, diag] = self.__output
        jacob_matrix -= self.__output[..., None] * self.__output[:, None]
        return output_gradient @ np.mean(jacob_matrix, axis=0)




data = load_iris()
encoder = OneHotEncoder(sparse_output=False)
Y = encoder.fit_transform(data["target"].reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(normalize(data["data"]), Y, test_size=0.8)

# y = y / np.max(y)
data = np.hstack((x_train, y_train))
batch_size = 1024
lr = 0.1
#
network = [Dense(4, 6), LeakyRelu(),
           Dense(6, 4), LeakyRelu(),
           Dense(4, 3), SoftMax()
           ]
#
for epoch in range(50000):
    num_batches = 0
    total_error = 0
    np.random.shuffle(data)
    # print(data)
    batches = np.array_split(data, range(batch_size, len(data), batch_size), axis=0)
    for batch in batches:
        output = batch[:, :4]
        target = batch[:, 4:]
        # print(np.hstack((output, target)))
        # break
        for layer in network:
            # print(data)
            output = layer.forward(output)
        #
        error, gradient = cat_cross_entropy(output, target)
        total_error += error
        num_batches += 1
        for layer in network[::-1]:
            gradient = layer.backward(gradient, lr)

        # print(f"{np.hstack((y, data))}")
        # print(f"Derivative: {derivative}")

    print(f"Epoch: {epoch}, Error: {total_error / num_batches}")

y_pred = x_test
for layer in network:
    y_pred = layer.forward(y_pred)

acc = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).sum() / len(y_pred)
print(acc)

# x = normalize(np.linspace(-100, 100, 1000).reshape(-1, 1))
# y = x
# for layer in network:
#     y = layer.forward(y)
# plt.plot(x, y)
# plt.show()
