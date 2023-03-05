import numpy as np


# Please replace "MatricNumber" with your actual matric number here and in the filename
def A1_A0233404U(X, A, y):
    """
    Input type
    :X type: numpy.ndarray
    :A type: numpy.ndarray
    :y type: numpy.ndarray

    Return type
    :InvXTAX type: numpy.ndarray
    :w type: numpy.ndarray
    """

    invXTAX = np.linalg.inv(X.T @ A @ X)
    w = invXTAX @ X.T @ A @ y
    return invXTAX, w


# A1=np.array([[1,0,0,0,0],[0,6,0,0,0],[0,0,7,0,0],[0,0,0,8,0],[0,0,0,0,3]])
# X1=np.array([[4,6],[0.1,6],[4,1],[0.7,6],[0.9,1]])
# y1=np.array([[2], [3], [4], [0.7], [2]])
# print (A1_A0233404U(X1, A1, y1))
