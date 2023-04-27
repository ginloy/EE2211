import numpy as np
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

DEPTH = 20

iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=0
)

classifier = DecisionTreeClassifier(criterion="entropy", max_depth=DEPTH)
classifier.fit(x_train, y_train)

print(classifier.score(x_train, y_train))
print(classifier.score(x_test, y_test))
print(classifier.predict(x_test))

plot_tree(classifier, filled=True)
plt.savefig(f"depth={DEPTH}.png", dpi=300)
