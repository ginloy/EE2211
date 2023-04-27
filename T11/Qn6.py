import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

model = KMeans(n_clusters=5)

center_1 = np.array([2, 2])
center_2 = np.array([4, 4])
center_3 = np.array([6, 1])

data_1 = np.random.randn(200, 2) + center_1
data_2 = np.random.randn(200, 2) + center_2
data_3 = np.random.randn(200, 2) + center_3

data = np.concatenate([data_1, data_2, data_3], axis=0)
labels = model.fit_predict(data)

plt.scatter(data[:, 0], data[:, 1], c=labels)
plt.show()

print(model.cluster_centers_)
