import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing as prep
import typing


def poly_reg(x: np.ndarray, y: np.ndarray, order: int, plot: bool = False) -> np.poly1d:
    if x.size != y.size:
        raise ValueError("Arrays are not of equal size")
    x = x.reshape(-1)
    y = y.reshape(-1)
    temp = np.polyfit(x, y, order)
    model = np.poly1d(temp)
    if not plot:
        return model
    points = np.arange(np.min(x) - 1, np.max(x) + 1, 0.5)
    res = model(points)
    plt.scatter(x, y)
    plt.plot(points, res)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()
    return model

def one_hot_encode(categories: np.ndarray) -> np.ndarray:
    encoder = prep.OneHotEncoder()
    return encoder.fit_transform(categories.reshape(-1, 1)).toarray()



x = np.array([-10, -8, -3, -1, 2, 8])
y = np.array([1, 1, 2, 1, 1, 2, 0, 1, 3, 0, 0])
print(one_hot_encode(y))