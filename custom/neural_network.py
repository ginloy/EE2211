from abc import abstractmethod
from typing import Tuple, Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle


def cat_cross_entropy(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    error = -np.mean(np.sum(target * np.log(pred), axis=1))
    gradient = (pred - target) / pred.shape[0]
    return error, gradient


def mse(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    error = np.mean((pred - target) ** 2, axis=0)
    gradient = (2 * (pred - target)) / len(pred)
    return error, gradient


def normalize(data: np.ndarray) -> np.ndarray:
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    std[std < 1e-20] = 1
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
        self.__input = inp
        return inp @ self.__weight + self.__bias

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        ret = output_gradient @ self.__weight.T
        self.__weight -= learn_rate * self.__input.T @ output_gradient
        self.__bias -= learn_rate * np.sum(output_gradient, axis=0, keepdims=True)
        return ret


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


class Relu(Layer):
    def __init__(self):
        self.__input = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        return np.maximum(inp, 0)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        temp = output_gradient.copy()
        temp[self.__input < 0] = 0
        return temp


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
        # batch_size = output_gradient.shape[0]
        # classes = output_gradient.shape[1]
        # eyes = np.tile(np.eye(classes), (batch_size, 1, 1))
        # jacob_matrix = np.einsum("ij,ijk->ijk", self.__output, eyes)\
        #     - np.einsum("ij,ik->ijk", self.__output, self.__output)
        # return np.einsum("ij,ijk->ik", output_gradient, jacob_matrix)
        return output_gradient


class Module:
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def optimize(self, gradients, lr):
        pass

    def fit(self, x: np.ndarray, y: np.ndarray, error_fn, lr, batch_size: int = 1, max_epochs: int = 10000):
        for epoch in range(max_epochs):
            x, y = shuffle(x, y)
            x_batches = np.array_split(x, range(batch_size, len(x), batch_size), axis=0)
            y_batches = np.array_split(y, range(batch_size, len(y), batch_size), axis=0)
            total_error = 0
            for batch_idx, x_batch in enumerate(x_batches):
                y_batch = y_batches[batch_idx]
                y_pred = self.forward(x_batch)
                loss = error_fn(y_pred, y_batch)
                total_error += loss[0]
                self.optimize(loss[1], lr)

            print(f"Epoch {epoch}, Error: {total_error / len(x_batches)}")

    def test(self, x: np.ndarray, y: np.ndarray, error_fn):
        return error_fn(self(x), y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class DigitModel(Module):
    def __init__(self):
        self.__layers = [
            Dense(64, 35), Relu(),
            Dense(35, 10), SoftMax()
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    def optimize(self, gradients, lr):
        for layer in self.__layers[::-1]:
            gradients = layer.backward(gradients, lr)

# data = load_digits()
# encoder = OneHotEncoder(sparse_output=False)
# Y = encoder.fit_transform(normalize(data["target"].reshape(-1, 1)))
#
# x_train, x_test, y_train, y_test = train_test_split(data["data"], Y, test_size=0.4)
#
# train_mean = np.mean(x_train)
# train_std = np.std(x_train)
#
# (x_train, x_test) = map(lambda x: (x - train_mean) / train_std, (x_train, x_test))

# model = DigitModel()
# model.fit(x_train, y_train, cat_cross_entropy, 0.1, batch_size=32, max_epochs=1000)

# y_pred = model(x_test)
# acc = (encoder.inverse_transform(y_pred) == encoder.inverse_transform(y_test)).sum() / len(y_pred)
# print(f"Accuracy: {acc}")
# x = np.linspace(-10, 10, 10000).reshape(-1, 1)
# y = x ** 3 + 2 * x**2 - 4 * x - 3

# x_train = normalize(x)
# y_train = normalize(y)
# y_train = y

# y = y / np.max(y)
# data = np.hstack((x_train, y_train))
# batch_size = 32
# lr = 0.1
# #
# network = [Dense(64, 35), Relu(),
#            Dense(35, 10), SoftMax()
#            ]
# #
# for epoch in range(1000):
#     num_batches = 0
#     total_error = 0
#     np.random.shuffle(data)
#     # print(data)
#     batches = np.array_split(data, range(batch_size, len(data), batch_size), axis=0)
#     for batch in batches:
#         output = batch[:, :64]
#         target = batch[:, 64:]
#         # print(np.hstack((output, target)))
#         # break
#         for layer in network:
#             # print(data)
#             output = layer.forward(output)
#         #
#         error, gradient = cat_cross_entropy(output, target)
#         total_error += error
#         num_batches += 1
#         for layer in network[::-1]:
#             gradient = layer.backward(gradient, lr)
#
#         # print(f"{np.hstack((y, data))}")
#         # print(f"Derivative: {derivative}")
#
#     print(f"Epoch: {epoch}, Error: {total_error / num_batches}")
#
# y_pred = x_test
# for layer in network:
#     y_pred = layer.forward(y_pred)
# #
# # plt.scatter(x_train, y_pred)
#
# acc = (np.argmax(y_pred, axis=1) == np.argmax(y_test, axis=1)).sum() / len(y_pred)
# print(acc)

# x = normalize(np.linspace(-100, 100, 1000).reshape(-1, 1))
# y = x
# for layer in network:
#     y = layer.forward(y)
# plt.plot(x, y)
# plt.show()
