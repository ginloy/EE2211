import numpy as np

points = np.array([0, 0, 0, 1, 1, 1, 1, 0, 3, 0, 3, 1, 4, 0, 4, 1]).reshape(-1, 2)

left = np.array([0, 0])
right = np.array([3, 0])

dist_left = np.sum((points - left) ** 2, axis=1, keepdims=True)
dist_right = np.sum((points - right) ** 2, axis=1, keepdims=True)
distances = np.hstack((dist_left, dist_right))

labels = np.argmin(distances, axis=1)
newleft = np.mean(points[labels == 0, :], axis=0)
newright = np.mean(points[labels == 1, :], axis=0)

print(newleft)
print(newright)
