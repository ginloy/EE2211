from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-100, 100, 1000).reshape(-1, 1)
y = np.sin(x)

model = DecisionTreeRegressor()
model.fit(x, y)

x_test = np.linspace(75, 150, 1000).reshape(-1, 1)
y_test = np.sin(x_test)
y_pred = model.predict(x_test)
plt.plot(x_test, y_test)
plt.scatter(x_test, y_pred)
plt.show()
plot_tree(model, filled=True)
plt.savefig("sin_test.png", dpi=1000)

