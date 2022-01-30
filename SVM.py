import pandas as pd
from IPython.display import display
import numpy as np
from numpy import float64
from sklearn.svm import SVC
from sklearn import metrics


def dataPreProcess():
    scores = pd.read_csv('malwareBenignScores.csv')
    scores = scores.drop('Sample', axis=1)

    Type = pd.Series([], dtype=float64)

    for i in range(len(scores)):
        if i < 40:
            Type[i] = 'malware'
        else:
            Type[i] = 'benign'
    scores.insert(3, 'Type', Type)

    X = scores.drop('Type', axis=1)
    Y = scores['Type']

    X_train = pd.concat([X[0:20], X[40:60]])
    Y_train = pd.concat([Y[0:20], Y[40:60]])

    X_test = pd.concat([X[20:40], X[60:80]])
    Y_test = pd.concat([Y[20:40], Y[60:80]])

    return X_train, Y_train, X_test, Y_test


def generalSVM():
    X_train, Y_train, X_test, Y_test = dataPreProcess()
    svm = SVC(kernel='poly', degree=4, C=3.0, gamma='auto')
    svm.fit(X_train, Y_train)
    y_pred = svm.predict(X_test)

    print("Accuracy for C=3, p=4:", metrics.accuracy_score(Y_test, y_pred))


def rbfSVM():
    X_train, Y_train, X_test, Y_test = dataPreProcess()
    # Index is C and columns are fi
    grid = pd.DataFrame(columns=['2', '3', '4', '5'], index=['1', '2', '3', '4'])

    for i in range(1, 5):
        for j in range(2, 6):
            svm = SVC(kernel='rbf', C=i, gamma=(1 / (2 * j ** 2)))
            svm.fit(X_train, Y_train)
            y_pred = svm.predict(X_test)
            grid[(str(j))][i - 1] = metrics.accuracy_score(Y_test, y_pred)
    print(grid)


def featureWeights():
    X_train, Y_train, X_test, Y_test = dataPreProcess()
    svm = SVC(kernel='linear', gamma='auto')
    svm.fit(X_train, Y_train)
    coef_dict = {}
    for coef, feat in zip(svm.coef_[0, :], X_train):
        coef_dict[feat] = coef
    print(coef_dict)

    X_train = X_train.drop([str(min(coef_dict, key=lambda y: abs(coef_dict[y])))], axis=1)
    svm = SVC(kernel='linear', gamma='auto')
    svm.fit(X_train, Y_train)
    coef_dict = {}
    for coef, feat in zip(svm.coef_[0, :], X_train):
        coef_dict[feat] = coef
    print(coef_dict)

    X_train = X_train.drop([str(min(coef_dict, key=lambda y: abs(coef_dict[y])))], axis=1)
    svm = SVC(kernel='linear', gamma='auto')
    svm.fit(X_train, Y_train)
    coef_dict = {}
    for coef, feat in zip(svm.coef_[0, :], X_train):
        coef_dict[feat] = coef
    print(coef_dict)

rbfSVM()