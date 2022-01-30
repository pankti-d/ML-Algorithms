import numpy as np

lm = [0, 0, 0, 0, 0, 0]
b = 0
epsilon = 0.00001
c = 2.5

X = [[3, 3], [3, 4], [2, 3], [1, 1], [1, 3], [2, 2]]
z = [1, 1, 1, -1, -1, -1]

data_dict = {
    1: [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)],
    2: [(1, 3), (2, 4), (3, 5), (4, 6)],
    3: [(1, 4), (2, 5), (3, 6)],
    4: [(1, 5), (2, 6)],
    5: [(1, 6)],
    6: [(2, 1), (3, 2), (4, 3), (5, 4), (6, 5)],
    7: [(3, 1), (4, 2), (5, 3), (6, 4)],
    8: [(4, 1), (5, 2), (6, 3)],
    9: [(5, 1), (6, 2)],
    10: [(6, 1)]
}


def F(P, lmval, zval, bval):
    Sum = 0
    for x in range(0, 6):
        Sum += lmval[x] * zval[x] * (np.dot(P, X[x]))
    Sum += bval
    return Sum


# l = 0
# h = 0
lmcap = [0, 0, 0, 0, 0, 0]

for m in range(0, 10):
    for k in range(1, 11):
        for n in range(0, len(data_dict[k])):
            i = data_dict[k][n][0] - 1
            j = data_dict[k][n][1] - 1
            d = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
            if abs(d) > epsilon:
                Ei = F(X[i], lm, z, b) - z[i]
                Ej = F(X[j], lm, z, b) - z[j]
                lmcap[i] = lm[i]
                lmcap[j] = lm[j]
                lm[j] = lm[j] - ((z[j] * (Ei - Ej)) / d)
                if z[i] == z[j]:
                    l = max(0, (lm[i] + lm[j] - c))
                    h = min(c, (lm[i] + lm[j]))
                else:
                    l = max(0, (lm[j] - lm[i]))
                    h = min(c, (c + lm[j] - lm[i]))
                if lm[j] > h:
                    lm[j] = h
                elif l <= lm[j] <= h:
                    lm[j] = lm[j]
                elif lm[j] < l:
                    lm[j] = l
                lm[i] = lm[i] + (z[i] * z[j] * (lmcap[j] - lm[j]))
                bi = b - Ei - z[i] * (lm[i] - lmcap[i]) * (np.dot(X[i], X[i])) - z[j] * (lm[j] - lmcap[j]) * (
                    np.dot(X[i], X[j]))
                bj = b - Ej - z[i] * (lm[i] - lmcap[i]) * (np.dot(X[i], X[j])) - z[j] * (lm[j] - lmcap[j]) * (
                    np.dot(X[j], X[j]))
                if 0 <= lm[i] <= c:
                    b = bi
                elif 0 <= lm[j] <= c:
                    b = bj
                else:
                    b = (bi + bj) / 2
print(lm)
