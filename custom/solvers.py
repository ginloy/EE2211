import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


class PolySolver:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, degree: int, bias: bool = True):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__transformer: PolynomialFeatures = PolynomialFeatures(degree, include_bias=bias)
        x_prepped = self.__transformer.fit_transform(self.__x_train)
        print(x_prepped)
        self.__weights = np.linalg.lstsq(x_prepped, self.__y_train, rcond=None)[0]

    def solve(self, x_test: np.ndarray) -> np.ndarray:
        if x_test.ndim != self.__x_train.ndim:
            raise ValueError("Test data has different dimensions as training data!")
        if x_test.shape[-1] != self.__x_train.shape[-1]:
            raise ValueError("Test data has different number of columns as training data!")
        x_prepped = self.__transformer.transform(x_test)
        y_test = x_prepped @ self.__weights
        return y_test

    def coefficients(self) -> np.ndarray:
        if self.__transformer.get_params()["include_bias"]:
            return self.__weights[1:, :]

    def intercept(self) -> float:
        return self.__weights[0, 0]

    def plot(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is None or y is None:
            x = self.__x_train
            y = self.__y_train
        if x.ndim > 2 or y.ndim > 2 or x.shape[-1] != 1 or y.shape[-1] != 1:
            raise ValueError("Only 2-dimensional data can be plotted!")
        points = np.linspace(np.min(x) - 1, np.max(x) + 1, 100).reshape((-1, 1))
        points_prepped = self.__transformer.transform(points)
        result = points_prepped @ self.__weights
        plt.scatter(x, y)
        plt.plot(points, result)
        plt.show()




# x = np.array([-10, -8, -3, -1, 2, 8]).reshape((-1, 1))
# y = np.array([5, 5, 4, 3, 2, 2]).reshape((-1, 1))
#
# solver = PolySolver(x, y, 3)
# solver.plot()
# print(solver.solve(x))