import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing as prep


def poly_reg(x: np.ndarray, y: np.ndarray, order: int, plot: bool = False) -> np.poly1d:
    if x.size != y.size:
        raise ValueError("Arrays are not of equal size")
    x = x.reshape(-1)
    y = y.reshape(-1)
    temp = np.polyfit(x, y, order)
    model = np.poly1d(temp)
    if not plot:
        return model
    points = np.linspace(np.min(x) - 1, np.max(x) + 1, 100)
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


def solve_system(x: np.ndarray, y: np.ndarray, bias=False) -> np.ndarray:
    """
    Solves system of equations
    :param x: (numpy matrix) Input data
    :param y: (numpy matrix) Target of equation
    :param bias: Whether to add a bias when doing regression
    :return: d x 1 numpy matrix of coefficients.
             If bias was set to true, additional bias coefficient at the top row
    """
    if len(x.shape) != 2 or len(y.shape) != 2:
        raise ValueError("Arrays are not matrices!")
    if bias:
        x = np.hstack((np.ones((x.shape[0], 1)), x))
    coeffs = np.linalg.lstsq(x, y, rcond=None)
    return coeffs[0]

# print(solve_system(np.array([1, 2, 3, 4]).reshape(2, 2), np.array([69, 70]).reshape(-1, 1), bias=True))
