# coding: utf-8

import matplotlib.pyplot as plt # plt 用于显示图片
import matplotlib.image as mpimg # mpimg 用于读取图片
import numpy as np
from sklearn.datasets import load_iris


def img():
    lena = mpimg.imread('http://scikit-learn.org/dev/_static/ml_map.png')
    plt.imshow(lena)
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def getData():
    iris = load_iris()
    n_samples, n_features = iris.data.shape
    print(iris.keys())
    print((n_samples, n_features))
    print(iris.data.shape)
    print(iris.target.shape)
    print(iris.target_names)
    print(iris.feature_names)


def show():
    iris = load_iris()
    # 'sepal width (cm)'
    x_index = 1
    # 'petal length (cm)'
    y_index = 2

    # this formatter will label the colorbar with the correct target names
    formatter = plt.FuncFormatter(lambda i, *args: iris.target_names[int(i)])

    plt.scatter(iris.data[:, x_index], iris.data[:, y_index],
                c=iris.target, cmap=plt.cm.get_cmap('RdYlBu', 3))
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.clim(-0.5, 2.5)
    plt.xlabel(iris.feature_names[x_index])
    plt.ylabel(iris.feature_names[y_index])
    plt.show()


show()