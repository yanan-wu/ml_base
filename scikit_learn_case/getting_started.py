# -*- coding: utf-8 -*

import mglearn
import pandas as pd
from pandas import plotting
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def train_model():
    # 获取数据
    iris_data = load_iris()

    # 拆分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(iris_data['data'], iris_data['target'], random_state=0)

    # 数据可视化，分析数据
    iris_dataframe = pd.DataFrame(X_train, columns=iris_data.feature_names)
    grr = plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15, 15), marker='o',
                            hist_kwds={'bins': 20}, s=60, alpha=.8, cmap=mglearn.cm3)

    # 训练模型
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    # 预测
    X_new = np.array([[5, 2.9, 1, 0.2]])
    prediction = knn.predict(X_new)
    print("Prediction: {}".format(prediction))
    print("Predicted target name: {}".format(
        iris_data['target_names'][prediction]))

    # 模型评估
    print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


if __name__ == "__main__":
    train_model()