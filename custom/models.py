import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder


class PolyModel:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, degree: int, bias: bool = True):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__transformer: PolynomialFeatures = PolynomialFeatures(degree, include_bias=bias)
        x_prepped = self.__transformer.fit_transform(self.__x_train)
        self.__weights = np.linalg.lstsq(x_prepped, self.__y_train, rcond=None)[0]

    def predict(self, x_test: np.ndarray) -> np.ndarray:
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
        return self.__weights

    def intercept(self) -> np.ndarray:
        if self.__transformer.get_params()["include_bias"]:
            return self.__weights[0, :]
        return np.zeros(self.__weights[0].shape)

    def plot(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is None or y is None:
            x = self.__x_train
            y = self.__y_train
        if x.ndim > 2 or y.ndim > 2 or x.shape[-1] != 1 or y.shape[-1] != 1:
            raise ValueError("Only 2-dimensional data can be plotted!")
        points = np.linspace(np.min(x) - 1, np.max(x) + 1, 100).reshape((-1, 1))
        result = self.predict(points)
        plt.scatter(x, y)
        plt.plot(points, result)
        plt.show()


class ClassificationModel:
    def __init__(self, x_train: np.ndarray, y_train: np.ndarray, power: int,  bias: bool = True):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__encoder = OneHotEncoder(sparse_output=False)
        y_transformed = self.__encoder.fit_transform(y_train)
        self.__poly_model = PolyModel(x_train, y_transformed, power, bias=bias)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        raw = self.__poly_model.predict(x_test)
        return self.__encoder.inverse_transform(raw)

    def coefficients(self) -> np.ndarray:
        return self.__poly_model.coefficients()

    def intercept(self) -> np.ndarray:
        return self.__poly_model.intercept()

    def plot(self, x: np.ndarray, y: np.ndarray = None):
        if y is None:
            y = self.predict(x)
        mapping = {}
        for i, category in enumerate(self.__encoder.categories_[0]):
            mapping[category] = i
        f = lambda cat: mapping[cat]
        color = np.vectorize(f)(y)
        df = pd.DataFrame(x)
        pd.plotting.scatter_matrix(df, c=color, marker='o', hist_kwds={"bins": 20})
        plt.show()


#
# wine_df = pd.read_csv("../T5/winequality-red.csv", sep=";")
# y = wine_df.quality.to_numpy().reshape(-1, 1)
# x = wine_df.drop("quality", axis=1).to_numpy()
# x_train, x_target, y_train, y_target = x[:1500, :], x[1500:, :], y[:1500, :], y[1500:, :]
# solver =PolyModel(x_train, y_train, 1)
# y_train_result = solver.predict(x_target)
# MSE = np.mean((y_train_result - y_target) ** 2)
# print(MSE)
# x = np.array([-10, -8, -3, -1, 2, 8]).reshape((-1, 1))
# y = np.array([5, 5, 4, 3, 2, 2]).reshape((-1, 1))
# PolyModel(x, y, 3).plot()
# x = np.array([-1, 0, 0.5, 0.3, 0.8]).reshape((-1, 1))
# y = np.array(["class1", "class1", "class2", "class1", "class2"]).reshape((-1, 1))
# model = ClassificationModel(x, y)
# print(model.predict(x))
# model.plot(x)

# data = load_iris()
# # %%
# x_train, x_test, y_train, y_test = train_test_split(data["data"], data["target"].reshape(-1, 1))
# model = ClassificationModel(x_train, y_train, 5)
# y_res = model.predict(x_train)
# print(y_res - y_train.reshape((-1, 1)))
