import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LR = 0.03
ITERS = int(2e6)

df = pd.read_csv("government-expenditure-on-education2.csv")

# print(df.columns)

max_year = df.year.max()
max_expenditure = df.total_expenditure_on_education.max()

df["normalized_year"] = df.year / max_year

df["normalized_expenditure"] = (
    df.total_expenditure_on_education / max_expenditure
)

x_train = np.array(df.normalized_year).reshape(-1, 1)
y_train = np.array(df.normalized_year).reshape(-1, 1)

test_years = np.arange(1981, 2024).reshape(-1, 1)
x_test = test_years / max_year

weights = np.random.rand(1, 1)
bias = np.zeros((1, 1))

iter = np.arange(1, ITERS + 1)
cost = []
for _ in range(ITERS):
    y_pred = np.exp(-(x_train @ weights + bias))
    loss = np.mean((y_pred - y_train) ** 2)
    cost.append(loss)
    loss_grad  = 2 * (y_pred - y_train) / len(y_train)
    out_grad = -loss_grad
    bias -= LR * np.sum(out_grad, axis=0, keepdims=True)
    weights -= LR * x_train.T @ out_grad

cost = np.array(cost)
plt.plot(iter, cost)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()

y_pred = np.exp(-(x_test @ weights + bias))
pred_expenditure = y_pred * max_expenditure

plt.plot(test_years, pred_expenditure)
plt.xlabel("Year")
plt.ylabel("Expenditure")
plt.show()

# print(weights, bias)
