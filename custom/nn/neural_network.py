from abc import abstractmethod
from typing import Tuple

import numpy as np
from numpy.lib.stride_tricks import as_strided
from sklearn.utils import shuffle


def cat_cross_entropy(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    error = -np.mean(np.sum(target * np.log(pred), axis=1))
    gradient = (pred - target) / len(pred)
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


class Convolution(Layer):
    def __init__(self, kernel_size: int, input_channels: int, output_channels: int):
        self.__input = None
        self.__kernel_size = kernel_size
        self.__kernels = np.random.normal(loc=0, scale=np.sqrt(2 / input_channels),
                                          size=(kernel_size, kernel_size, input_channels, output_channels))
        self.__bias = np.zeros((1, 1, 1, output_channels))

    @classmethod
    def correlate(cls, a: np.ndarray, b: np.ndarray):
        temp = cls.windows(a, (b.shape[0], b.shape[1]))
        return np.tensordot(temp, b, axes=3)

    @classmethod
    def windows(cls, a: np.ndarray, window_shape) -> np.ndarray:
        hout = a.shape[1] - window_shape[0] + 1
        wout = a.shape[2] - window_shape[1] + 1
        return as_strided(a, (a.shape[0], hout, wout, window_shape[0], window_shape[1], a.shape[3]),
                          a.strides[:3] + a.strides[1:])

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        return self.correlate(inp, self.__kernels) + self.__bias

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        bias_grads = np.sum(output_gradient, axis=(0, 1, 2))

        pad_amt = self.__kernel_size - 1
        padded_grads = np.pad(output_gradient, ((0, 0), (pad_amt, pad_amt), (pad_amt, pad_amt), (0, 0)),
                              mode="constant", constant_values=0)
        flipped_kernels = self.__kernels[::-1, ::-1].transpose(0, 1, 3, 2)
        input_grads = self.correlate(padded_grads, flipped_kernels)

        input_windows = self.windows(self.__input, (output_gradient.shape[1], output_gradient.shape[2]))
        kernel_grads = np.tensordot(input_windows, output_gradient, axes=((0, 3, 4), (0, 1, 2)))

        self.__kernels -= learn_rate * kernel_grads
        self.__bias -= learn_rate * bias_grads
        return input_grads


class MaxPool(Layer):
    def __init__(self):
        self.__mask = None
        self.__inp_shape = None
        self.__padded_shape = None

    @classmethod
    def windows(cls, inp: np.ndarray, window_shape) -> np.ndarray:
        hout = inp.shape[1] // window_shape[0]
        wout = inp.shape[2] // window_shape[1]
        return as_strided(inp, (inp.shape[0], hout, wout, window_shape[0], window_shape[1], inp.shape[3]),
                          inp.strides[:1] + (
                              inp.strides[1] * window_shape[0], inp.strides[2] * window_shape[1]) + inp.strides[1:])

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__inp_shape = inp.shape
        if self.__inp_shape[1] % 2 == 1 and self.__inp_shape[2] % 2 == 1:
            inp = np.pad(inp, ((0, 0), (0, 1), (0, 1), (0, 0)), mode="constant", constant_values=0)
        elif self.__inp_shape[1] % 2 == 1:
            inp = np.pad(inp, ((0, 0), (0, 1), (0, 0), (0, 0)), mode="constant", constant_values=0)
        elif self.__inp_shape[2] % 2 == 1:
            inp = np.pad(inp, ((0, 0), (0, 0), (0, 1), (0, 0)), mode="constant", constant_values=0)
        self.__padded_shape = inp.shape
        # windows = sliding_window_view(inp, (2, 2), axis=(1, 2))[:, ::2, ::2]
        windows = self.windows(inp, (2, 2))
        mx = windows.max(axis=(3, 4), keepdims=True)
        self.__mask = np.isclose(windows, mx)
        return mx.squeeze()

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        windows = self.windows(output_gradient, (1, 1))
        temp = (self.__mask * windows).transpose(0, 1, 3, 2, 4, 5).reshape(self.__padded_shape)
        if self.__inp_shape[1] % 2 == 1 and self.__inp_shape[2] % 2 == 1:
            return temp[:, :-1, :-1, :]
        elif self.__inp_shape[1] % 2 == 1:
            return temp[:, :-1, :, :]
        elif self.__inp_shape[2] % 2 == 1:
            return temp[:, :, :-1, :]
        return temp


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


class Flatten(Layer):
    def __init__(self):
        self.__input_shape = None

    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input_shape = inp.shape
        return inp.reshape(len(inp), -1)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        return output_gradient.reshape(self.__input_shape)


class Module:
    def __init__(self):
        self.normalize = lambda x: x

    @abstractmethod
    def _forward_raw_data(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def optimize(self, gradients, lr):
        pass

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._forward_raw_data(self.normalize(x))

    def fit(self, x: np.ndarray, y: np.ndarray, error_fn, lr, batch_size: int = 1, max_epochs: int = 10000):
        dims = x.ndim
        axes = tuple(range(dims - 1))
        train_mean = np.mean(x, axis=axes)
        train_std = np.std(x, axis=axes)
        train_std[train_std == 0] = 1
        self.normalize = lambda x_raw: (x_raw - train_mean) / train_std
        x = self.normalize(x)
        for epoch in range(max_epochs):
            x, y = shuffle(x, y)
            x_batches = np.array_split(x, range(batch_size, len(x), batch_size), axis=0)
            y_batches = np.array_split(y, range(batch_size, len(y), batch_size), axis=0)
            total_error = 0
            for batch_idx, x_batch in enumerate(x_batches):
                y_batch = y_batches[batch_idx]
                y_pred = self._forward_raw_data(x_batch)
                loss = error_fn(y_pred, y_batch)
                total_error += loss[0]
                self.optimize(loss[1], lr)

            print(f"\rEpoch {epoch}, Error: {total_error / len(x_batches)}", end="")
        print()

    def test(self, x: np.ndarray, y: np.ndarray, error_fn):
        return error_fn(self(x), y)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class DigitModel(Module):
    def __init__(self):
        super().__init__()
        self.__layers = [
            Dense(64, 35), Relu(),
            Dense(35, 10), SoftMax()
        ]

    def _forward_raw_data(self, x: np.ndarray) -> np.ndarray:
        for layer in self.__layers:
            x = layer.forward(x)
        return x

    def optimize(self, gradients, lr):
        for layer in self.__layers[::-1]:
            gradients = layer.backward(gradients, lr)

# data = load_digits()
# encoder = OneHotEncoder(sparse_output=False)
# Y = encoder.fit_transform(data["target"].reshape(-1, 1))
#
# x_train, x_test, y_train, y_test = train_test_split(data["data"], Y, test_size=0.4)
#
# # print(x_train, x_test)
#
# model = DigitModel()
# model.fit(x_train, y_train, cat_cross_entropy, 0.1, batch_size=32, max_epochs=1000)
#
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
