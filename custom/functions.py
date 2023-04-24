import numpy as np

def gini_impurity(lst: list[float]) -> float:
    temp = 1
    for i in lst:
        temp -= i**2
    return temp

def entropy(lst: list[float]) -> float:
    temp = 0
    for i in lst:
        temp -= i*np.log2(i)
    return temp

