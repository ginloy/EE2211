import numpy as np

def A1_A0233515M(X, A, y):
    Xtranspose = X.T
    InvXTAX = np.linalg.inv(Xtranspose@A@X)
    w = InvXTAX@Xtranspose@A@y
    return InvXTAX, w
   
A1=np.array([[1,0,0,0,0],[0,6,0,0,0],[0,0,7,0,0],[0,0,0,8,0],[0,0,0,0,3]])
X1=np.array([[4,6],[0.1,6],[4,1],[0.7,6],[0.9,1]])
y1=np.array([[2], [3], [4], [0.7], [2]])
print (A1_A0233515M(X1, A1, y1))

