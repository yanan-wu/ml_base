# coding: utf-8

from sklearn import tree
from sklearn.datasets import load_iris
import numpy as np


def classifier():
    features = [[140, 1], [130, 1], [150, 0], [160, 0]]
    lables = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, lables)
    print(clf.predict([[150, 0]]))


def tree_decision():
    iris = load_iris()
    test_idx = [0, 50, 100]
    train_target = np.delete(iris.target, test_idx)
    train_data = np.delete(iris.data, test_idx, axis=0)

    test_target = iris.target[test_idx]
    test_data = iris.data[test_idx]

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)

    print(test_target)
    print(clf.predict(test_data))


tree_decision()