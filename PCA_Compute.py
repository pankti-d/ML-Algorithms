import numpy as np
from numpy import ndarray

B = np.matrix([[2, -2, -1, 3], [-1, 3, 3, -1], [0, 2, 3, 0], [1, 3, 1, 3], [1, 0, -1, 2], [-3, 2, 4, -1], [5, -1, 5, 3],
               [2, 1, 2, 0]])

myu = B.sum(axis=1) / 4

A = np.matrix(B - myu)

C = 0.25 * np.dot(A, A.T)

U, S, V = np.linalg.svd(A)

eigvalue, eigvec = np.linalg.eig(C)

U = ndarray.round(U[:, :3], 4)

print('\nTraining Matrix B\n', B)
print('\nNormalised Matrix A\n', A)
print('Highest Eigenvalues for covariance C\n', ndarray.round(eigvalue[eigvalue > 1], 4))
print('\nCorresponding Unit Vectors\n', U)


delta = np.matrix(np.dot(U.T, A))
print('\nScoring Matrix delta\n', delta)

y1 = np.matrix([1, 5, 1, 5, 5, 1, 1, 3]).T
y2 = np.matrix([-2, 3, 2, 3, 0, 2, -1, 1]).T
y3 = np.matrix([2, -3, 2, 3, 0, 0, 2, -1]).T
y4 = np.matrix([2, -2, 2, 2, -1, 1, 2, 2]).T

y1 = y1 - myu
w1 = np.dot(U.T, y1)

y2 = y2 - myu
w2 = np.dot(U.T, y2)

y3 = y3 - myu
w3 = np.dot(U.T, y3)

y4 = y4 - myu
w4 = np.dot(U.T, y4)

min_score = []
for col in delta.T:
    dist = np.linalg.norm(w1 - col.T)
    min_score.append(dist)
print("Min Score Y1:\n", np.amin(min_score))

min_score.clear()
for col in delta.T:
    dist = np.linalg.norm(w2 - col.T)
    min_score.append(dist)
print("\nMin Score Y2:\n", np.amin(min_score))

min_score.clear()
for col in delta.T:
    dist = np.linalg.norm(w3 - col.T)
    min_score.append(dist)
print("\nMin Score Y3:\n", np.amin(min_score))

min_score.clear()
for col in delta.T:
    dist = np.linalg.norm(w4 - col.T)
    min_score.append(dist)
print("\nMin Score Y4:\n", np.amin(min_score))
