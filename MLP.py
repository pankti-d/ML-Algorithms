w = [1, 2, -1, 1, -2, 1]
alpha = 0.1
train = [(0.6, 0.4, 1), (0.1, 0.2, 0), (0.8, 0.6, 0), (0.3, 0.7, 1), (0.7, 0.3, 1), (0.7, 0.7, 0), (0.2, 0.9, 1)]
test = [(0.55, 0.11, 1), (0.32, 0.21, 0), (0.24, 0.64, 1), (0.86, 0.68, 0), (0.53, 0.79, 0), (0.46, 0.54, 1),
        (0.16, 0.51, 1), (0.52, 0.94, 0), (0.46, 0.87, 1), (0.96, 0.63, 0)]
epsilon = 2.71828
vi = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dv = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
n = 0
# Forward Pass
for j in range(0, 1000):
    for i in range(0, len(train)):
        vi[0] = w[0]
        vi[1] = w[1]
        vi[2] = w[2]
        vi[3] = w[3]
        vi[4] = w[4]
        vi[5] = w[5]
        vi[6] = train[i][0] * vi[0] + train[i][1] * vi[2]
        vi[7] = train[i][0] * vi[1] + train[i][1] * vi[3]
        vi[8] = 1 + epsilon ** (-vi[6])
        vi[9] = 1 + epsilon ** (-vi[7])
        vi[10] = vi[4] / vi[8]
        vi[11] = vi[5] / vi[9]
        vi[12] = (vi[10] + vi[11] - train[i][2]) ** 2 / 2
        # Backward Pass
        dv[12] = 1
        dv[11] = vi[10] + vi[11] - train[i][2]
        dv[10] = vi[10] + vi[11] - train[i][2]
        dv[9] = (-vi[5] / vi[9] ** 2) * dv[11]
        dv[8] = (-vi[4] / vi[8] ** 2) * dv[10]
        dv[7] = (-epsilon ** (-vi[7])) * dv[9]
        dv[6] = (-epsilon ** (-vi[6])) * dv[8]
        dv[5] = dv[11] / vi[9]
        dv[4] = dv[10] / vi[8]
        dv[3] = train[i][1] * dv[7]
        dv[2] = train[i][1] * dv[6]
        dv[1] = train[i][0] * dv[7]
        dv[0] = train[i][0] * dv[6]
        for k in range(0, len(w)):
            w[k] = w[k] - alpha * dv[k]
    j = j + 1


def f(val1, val2):
    ans = 1 / (1 + epsilon ** -(val1 + val2))
    return ans


def scoring(w_, train_):
    count = 0
    Y = [0] * len(train_)
    for i in range(0, len(Y)):
        Y[i] = w_[4] * f(w_[0] * train_[i][0], w_[2] * train_[i][1]) + w_[5] * f(w_[1] * train_[i][0],
                                                                                 w_[3] * train_[i][1])
        if Y[i] < 0.5:
            Y[i] = 0
        else:
            Y[i] = 1
        if Y[i] == train_[i][2]:
            count += 1
    return count, Y


print("For 10000 Epochs:\n")

print("Weights after 10000 epochs:\t", w)

count, Y = scoring(w, train)
print("XOR Output for Training Data:\t", Y)
print("Accuracy for training data:\t", count / len(train))

count, Y = scoring(w, test)
print("XOR Output for Test Data:\t", Y)
print("Accuracy for Test data:\t", count / len(test))
