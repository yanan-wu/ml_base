# coding: utf-8

from sklearn import tree


def classifier():
    features = [[140, 1], [130, 1], [150, 0], [160, 0]]
    lables = [0, 0, 1, 1]
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(features, lables)
    print(clf.predict([[150, 0]]))

classifier()