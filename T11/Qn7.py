from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd

data = load_iris()
x = data.data
y = data.target

model = KMeans(3)

predicted_labels = model.fit_predict(x)
df = pd.DataFrame(x, columns=data.feature_names)
pd.plotting.scatter_matrix(df, c=y)
plt.savefig("actual.png")
plt.clf()
pd.plotting.scatter_matrix(df, c=predicted_labels)
plt.savefig("predicted.png")


