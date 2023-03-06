# from sklearn.model_selection import train_test_split
# from sklearn.datasets import load_iris
from abc import abstractmethod
from typing import Union

import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from matplotlib import pyplot as plt
# import copy
#
# from torch.utils.data import DataLoader, TensorDataset
# from torchmetrics import Accuracy

from custom.models import *


def cat_cross_entropy(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = np.mean(np.sum(-target * np.log(np.clip(pred, 1e-7, 1 - 1e-7)), axis=1))
    derivative = np.mean(pred - target, axis=0, keepdims=True)
    return error, derivative


def mse(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    error = np.mean((pred - target) ** 2)
    gradient = np.mean(2 * (pred - target), axis=0, keepdims=True)
    return error, gradient


class Layer:
    def __init__(self):
        self.__input: Union[np.ndarray, None] = None

    @abstractmethod
    def forward(self, inp: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        pass


class Dense(Layer):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.__weight = np.random.normal(loc=0, scale=np.sqrt(1/inp_dim), size=(inp_dim, out_dim))
        self.__bias = np.zeros((1, out_dim))

    def forward(self, inp: np.ndarray) -> np.ndarray:
        # print(self.__weight)
        self.__input = inp
        return inp @ self.__weight + self.__bias

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        mean_input = np.mean(self.__input, axis=0, keepdims=True)
        # print(output_gradient, mean_input)
        prev_weight = self.__weight[:, :]
        self.__weight -= learn_rate * output_gradient * mean_input.T
        self.__bias -= learn_rate * output_gradient
        return output_gradient @ prev_weight.T


class LeakyRelu(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        return np.clip(inp, 0, 1)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        temp = np.ones(self.__input.shape)
        temp[temp < 0] = 0
        return np.mean(gradient, axis=0, keepdims=True) * output_gradient
        # return learn_rate * (output_gradient > 0)

class Tanh(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        return np.tanh(inp)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        return np.mean((1 - np.tanh(self.__input) ** 2) * output_gradient, axis=0, keepdims=True)

class SoftMax(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        self.__input = inp
        prep = inp - np.max(inp, axis=1, keepdims=True)
        return np.exp(prep) / np.sum(prep, axis=1, keepdims=True)

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        return output_gradient


class Linear(Layer):
    def forward(self, inp: np.ndarray) -> np.ndarray:
        return inp

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        return output_gradient


x = ((np.random.rand(1000)-0.5) * 4 * np.pi).reshape(-1, 1)
y = x ** 3 - (2 * x**2) + 5
# y = y / np.max(y)
data = np.hstack((x, y))
batch_size = 1
lr = 0.001
#
network = [Dense(1, 64),Tanh(), Dense(64, 64),Tanh(),Dense(64, 1)]
#
for epoch in range(150):
    batches = 0
    total_error = 0
    np.random.shuffle(data)
    # print(data)
    for i in range(0, len(data), batch_size):
        output = data[i:i + batch_size, :1]
        target = data[i:i + batch_size, 1:2]
        # print(np.hstack((output, target)))
        # break
        for layer in network:
            # print(data)
            output = layer.forward(output)
        #
        error, gradient = mse(output, target)
        total_error += error
        batches += 1
        for layer in network[::-1]:
            gradient = layer.backward(gradient, lr)

        # print(f"{np.hstack((y, data))}")
        # print(f"Derivative: {derivative}")

    print(f"Epoch: {epoch}, Error: {total_error / batches}")

x_test = np.linspace(-3 * np.pi, 3 * np.pi, 100).reshape(-1, 1)
y_pred = x_test
for layer in network:
    y_pred = layer.forward(y_pred)
# plt.ylim(0, 5)
# print(y_pred)
plt.plot(x_test, y_pred)
plt.scatter(x, y)
plt.show()
# y_pred = x
# for layer in network:
#     y_pred = layer.forward(y_pred)
#
# # model = PolyModel(x, y, 20)
# #
# # # print(x)
# # plt.scatter(x, y)
# # x_test = np.linspace(-20, 20).reshape(-1, 1)
# # plt.ylim(np.min(y), np.max(y))
# # plt.xlim(np.min(x), np.max(x))
# plt.scatter(x, y_pred)
# plt.ylim(0, 10)
# plt.show()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # device = torch.device("cpu")
#
#
# model = nn.Sequential(
#     nn.Linear(4, 256),
#     nn.ReLU(inplace=True),
#     nn.Linear(256, 128),
#     nn.ReLU(inplace=True),
#     nn.Linear(128, 64),
#     nn.ReLU(inplace=True),
#     nn.Linear(64, 32),
#     nn.ReLU(inplace=True),
#     nn.Linear(32, 3),
#     nn.Softmax(dim=1)
# ).to(device)
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0001)
# batch_size = 1024
# epochs = 1000
#
# temp = ((np.random.rand(10000) - 0.5) * 6 * np.pi).reshape(-1, 1)
# tempres = np.sin(temp) * 5
# data = load_iris()
# x = data["data"]
# y = data["target"].reshape(-1, 1)
# x_train, x_test, y_train, y_test = map(lambda x: torch.tensor(x, dtype=torch.float32), train_test_split(x, y, test_size=0.2))
# dataset = TensorDataset(x_train, torch.squeeze(y_train.type(torch.LongTensor)))
# test_dataset = TensorDataset(x_test, torch.squeeze(y_test.type(torch.LongTensor)))
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
# test_dataloader = DataLoader(test_dataset)
#
# best_mse = np.inf
# best_weights = None
# history = []
#
# for epoch in range(epochs):
#     print(f"Epoch {epoch}/{epochs}")
#     model.train()
#     for x, y in dataloader:
#         x = x.to(device)
#         y = y.to(device)
#
#         optimizer.zero_grad()
#
#         score = model(x)
#         loss = loss_fn(input=score, target=y)
#         loss.backward()
#
#         optimizer.step()
#
#     model.eval()
#     for x, y in dataloader:
#         x = x.to(device)
#         y = y.to(device)
#
#         y_pred = model(x)
#         mse = loss_fn(y_pred, y)
#         mse = float(mse)
#         history.append(mse)
#         if mse < best_mse:
#             best_mse = mse
#             best_weights = copy.deepcopy(model.state_dict())
#
# # plt.plot(x.clone().detach().numpy(), model(x).clone().detach().numpy())
# # plt.show()
#
# model.load_state_dict(best_weights)
# torch.save(model, "iris.ser")
# plt.plot(history)
# plt.show()
#
# model.to(torch.device("cpu"))
# model.eval()
# y_pred = torch.argmax(model(x_test), dim=1, keepdim=True)
# accuracy = Accuracy(task="multiclass", num_classes=3)
# print(accuracy(y_pred, y_test))
#
#
# # points = torch.linspace(-5 * torch.pi, 5 * torch.pi, 100).view(-1, 1).to(device)
# # res = model(points)
#
# # plt.scatter(temp, tempres)
# # plt.plot(points.cpu().detach().numpy(), res.cpu().detach().numpy())
# # plt.ylim(-10, 10)
# # plt.show()
# # plt.scatter(x.numpy(), model(x).clone().detach().numpy())
# # plt.show()
