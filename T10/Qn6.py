from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures

data = load_iris()
# print(data.keys())
x = data.data
y = data.target.reshape(-1, 1)

x_train, x_test, y_train, t_test = train_test_split(x, y, test_size=0.2)

encoder = OneHotEncoder(sparse=False)
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.fit(t_test)

best_score = -1e9
best_degree = 0
for degree in range(1, 11):
    model = Pipeline(
        [
            (f"poly{degree}", PolynomialFeatures(degree)),
            ("linear", LinearRegression(fit_intercept=False)),
        ],
    )
    res = cross_validate(
        model,
        x_train,
        y_train_encoded,
        cv=5,
        scoring="neg_mean_squared_error",
        return_estimator=True,
    )
    average_test_score = res["test_score"].mean()
    if average_test_score > best_score:
        best_score = average_test_score
        best_degree = degree
        best_model = res["estimator"][0]
    print(res)

print(best_degree, best_score, best_model)
