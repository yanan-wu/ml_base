# coding: utf-8

from scipy.spatial import distance
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def process():
    iris = load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

    my_clf = ScrappyKNN()
    my_clf.fit(X_train, y_train)
    predictions = my_clf.predict(X_test)
    # 实际预测得分
    print(accuracy_score(y_test, predictions))


def ecu(a, b):
    return distance.euclidean(a, b)


# 自定义分类器
class ScrappyKNN():
    def fit(self, X_train, y_trian):
        self.X_trian = X_train
        self.y_train = y_trian

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = ecu(row, self.X_trian[0])
        best_index = 0
        for i, item in enumerate(self.X_trian):
            dist = ecu(row, item)
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


process()