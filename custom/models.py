from typing import Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures


class PolyModel:
    """
    An immutable polynomial regression model fitted to the given data

    Parameters:
       x_train (np.ndarray): The training input
       y_train (np.ndarray): The training output
       degree (int): The degree of the polynomial
       bias (bool): Whether to include a bias term
       ridge (float): The ridge parameter (if 0, no ridge is used)
    """
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        degree: int,
        bias=True,
        ridge=0.0,
    ):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__transformer: PolynomialFeatures = PolynomialFeatures(
            degree, include_bias=bias
        )
        self.__linear_model = (
            LinearRegression(fit_intercept=False)
            if abs(ridge) < 1e-9
            else Ridge(ridge, fit_intercept=False)
        )
        x_prepped = self.__transformer.fit_transform(self.__x_train)
        self.__linear_model.fit(x_prepped, y_train)

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        """
        Predicts the output for the given input

        Parameters:
            x_test (np.ndarray): The input to predict the output for

        Returns:
            np.ndarray: The predicted output
        """
        if x_test.ndim != self.__x_train.ndim:
            raise ValueError("Test data has different dimensions as training data!")
        if x_test.shape[-1] != self.__x_train.shape[-1]:
            raise ValueError(
                "Test data has different number of columns as training data!"
            )
        x_prepped = self.__transformer.transform(x_test)
        y_test = self.__linear_model.predict(x_prepped)
        return y_test

    def coefficients(self) -> np.ndarray:
        return self.__linear_model.coef_

    def plot(self, x: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None):
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

    def transform(self, x: np.ndarray):
        return self.__transformer.transform(x)

    @classmethod
    def unfitted_transform(cls, x: np.ndarray, power: int, bias=True) -> np.ndarray:
        return PolynomialFeatures(power, include_bias=bias).fit_transform(x)


class ClassificationModel:
    """
    An immutable classification model fitted to the given data

    :param x_train: The training input
    :param y_train: The training labels
    :param power: The degree of the polynomial
    :param bias: Whether to include a bias term
    :param ridge: The ridge parameter (if 0, no ridge is used)
    """

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        power: int,
        bias: bool = True,
        ridge: float = 0.0,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.__encoder = OneHotEncoder(sparse=False)
        y_transformed = self.__encoder.fit_transform(y_train)
        self.__poly_model = PolyModel(
            x_train, y_transformed, power, bias=bias, ridge=ridge
        )

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        raw = self.__poly_model.predict(x_test)
        return self.__encoder.inverse_transform(raw)

    def coefficients(self) -> np.ndarray:
        return self.__poly_model.coefficients()

    def categories(self) -> np.ndarray:
        return self.__encoder.categories_[0]

    def plot(self, x: np.ndarray, y: Optional[np.ndarray] = None):
        if y is None:
            y = self.predict(x)
        mapping = {}
        for i, category in enumerate(self.categories()):
            mapping[category] = i

        def f(cat):
            return mapping[cat]

        color = np.vectorize(f)(y)
        df = pd.DataFrame(x)
        pd.plotting.scatter_matrix(df, c=color, marker="o", hist_kwds={"bins": 20})
        plt.show()

    def transform(self, y: np.ndarray):
        return self.__encoder.transform(y)

    @classmethod
    def one_hot_encode(cls, y: np.ndarray) -> np.ndarray:
        return OneHotEncoder(sparse=False).fit_transform(y)
