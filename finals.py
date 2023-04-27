import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from custom.models import PolyModel, ClassificationModel
from T10.Qn6 import PolynomialClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

def qn3():
    y_pred = np.array([7, 7, 9, 9])
    y = 8.5
    mean = np.mean(y_pred)
    variance = np.var(y_pred)
    mse = np.mean((y_pred - y) ** 2)
    bias_squared = (y - np.mean(y_pred))**2
    noise = mse - bias_squared - variance
    print(mean, variance, mse, bias_squared, noise)

# qn3()

def qn4():
    w = np.array([1.0, 2.0]).reshape(-1, 1)
    for iter in range(1, 1000):
        gradient = np.array([w[0, 0] + 0.5 * (w[1, 0]**2), 0.5 * (w[0, 0] ** 2) + w[1, 0] ]).reshape(-1, 1)
        w -= 0.1 * gradient
        if (np.max(w) < 1e-10):
            print(iter)
            break

# qn4()

def qn5():
    x = np.array([-2, -2, -1, 0, 1]).reshape(-1, 1)
    y = np.array([8, 9, 5, 2, -1]).reshape(-1, 1)
    data = np.hstack((x, y))
    for threshold in [-1.5, -0.5, 0.5]:
        left = data[data[:, 0] < threshold]
        right = data[data[:, 0] >= threshold]
        left_mean = np.mean(left[:, 1])
        right_mean = np.mean(right[:, 1])
        overall_mse = (np.mean((left[:, 1] - left_mean)**2) + np.mean((right[:, 1] - right_mean) ** 2)) / 2
        print(overall_mse)


# qn5()


def qn28():
    centroids = np.array([100, 85, 66]).reshape(-1, 1)
    marks = np.array([60, 66, 80, 85, 90, 100])
    # distances = (marks - centroids) ** 2
    # labels = np.argmin(distances, axis=0)
    # print(labels)
    marks = np.array([80, 70, 72, 74, 85, 92, 98, 100]).reshape(-1, 1)
    classifier = KMeans(3,init=np.array([100, 80, 70]).reshape(-1, 1))
    classifier.fit(marks)
    print(classifier.predict(marks))

# DecisionTreeClassifier()

# qn28()

def qn32():
    total = 0
    valid = 0
    set = [1, -1]
    for i in set:
        for j in set:
            for k in set:
                for l in set:
                    total += 1
                    matrix = np.array([[i, j], [k, l]])
                    if np.linalg.matrix_rank(matrix) == 2:
                        print(np.linalg.matrix_rank(matrix))
                        valid += 1
    print(valid / total)

qn32()

