#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys

LR = float(sys.argv[1])
ITERS = int(2e6)

df = pd.read_csv("government-expenditure-on-education2.csv")

# print(df.columns)

max_year = df.year.max()
max_expenditure = df.total_expenditure_on_education.max()

df["normalized_year"] = df.year / max_year

df["normalized_expenditure"] = df.total_expenditure_on_education / max_expenditure

x_train = np.array(df.normalized_year).reshape(-1, 1)
x_train = np.hstack((np.ones_like(x_train), x_train))
y_train = np.array(df.normalized_expenditure).reshape(-1, 1)

test_years = np.arange(1981, 2024).reshape(-1, 1)
x_test = test_years / max_year
x_test = np.hstack((np.ones_like(x_test), x_test))

weights = np.zeros((2, 1))

iter = np.arange(1, ITERS + 1)
cost = []
for epoch in range(ITERS):
    y_pred = np.exp(-(x_train @ weights))
    loss = np.mean((y_pred - y_train) ** 2)
    cost.append(loss)
    loss_grad = 2 * (y_pred - y_train) / y_pred.shape[0]
    out_grad = -loss_grad * y_pred
    weights -= LR * x_train.T @ out_grad

    if epoch % int(2e5) == 0:
        print(f"Epoch {epoch}:\tloss = {loss}")

cwd = os.path.curdir
dir = f"{cwd}/Figures"
if not os.path.isdir(dir):
    os.mkdir(dir)

cost = np.array(cost)
plt.plot(iter, cost)
plt.title(f"Learning Rate = {LR}")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.savefig(f"{dir}/cost_lr_{LR}.png")
plt.clf()

y_pred = np.exp(-(x_test @ weights))
pred_expenditure = y_pred * max_expenditure

plt.plot(test_years, pred_expenditure)
plt.scatter(df.year, df.total_expenditure_on_education)
plt.title(f"Learning Rate = {LR}")
plt.xlabel("Year")
plt.ylabel("Expenditure")
plt.savefig(f"{dir}/pred_lr_{LR}.png")

# print(weights, bias)
