# -*- coding: utf-8 -*

from sklearn.datasets import load_iris


def get_iris_data():
    iris_data = load_iris()
    print(iris_data)


if __name__ == "__main__":
    get_iris_data()