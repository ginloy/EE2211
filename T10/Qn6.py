import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from sklearn.utils.multiclass import unique_labels

data = load_iris()
# print(data.keys())
x = data.data
y = data.target

x_train, x_test, y_train, t_test = train_test_split(x, y, test_size=0.2)


class PolynomialClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, power: int, bias=True, ridge=0.0):
        self.power = power
        self.bias = bias
        self.ridge = ridge

    def fit(self, x, y):
        x, y = check_X_y(x, y)
        self.classes_ = unique_labels(y)
        y = y.reshape(-1, 1)
        self.encoder_ = OneHotEncoder(sparse_output=False)
        y_encoded = self.encoder_.fit_transform(y)
        self.poly_ = PolynomialFeatures(self.power, include_bias=self.bias)
        p = self.poly_.fit_transform(x)
        self.linear_ = (
            Ridge(self.ridge, fit_intercept=False)
            if self.ridge > 1e-9
            else LinearRegression(fit_intercept=False)
        )
        self.linear_.fit(p, y_encoded)

    def predict(self, x):
        check_is_fitted(self)
        x = check_array(x)
        p = self.poly_.transform(x)
        y_encoded = self.linear_.predict(p)
        return self.encoder_.inverse_transform(y_encoded)


# print(check_estimator(Model(2)))


best_score = -1e9
best_degree = 0
best_model = None
ridges = np.linspace(0, 1, 10)
for degree in range(1, 11):
    model = PolynomialClassifier(degree, ridge=0)
    res = cross_validate(
        model,
        x_train,
        y_train,
        cv=5,
        scoring="accuracy",
        return_estimator=True,
    )
    average_test_score = res["test_score"].mean()
    if average_test_score > best_score:
        best_score = average_test_score
        best_degree = degree
        best_model = res["estimator"][0]
    # print(res)

print(best_degree, best_score)
print(best_model.get_params())
print(best_model.classes_)
