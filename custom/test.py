from models import PolyModel
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


x = np.array([4, 7, 9, 2, 3, 10]).reshape(-1, 1)
y = np.array([-1, -1, -1, 1, 1, 1]).reshape(-1, 1)

model = PolyModel(x, y, 4)
model.plot()

x_test = np.array([6]).reshape(-1, 1)
print(model.predict(x))
print(model.coefficients())
print(model.transform(x_test))
print(model.predict(x_test))


