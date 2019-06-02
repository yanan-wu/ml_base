# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture


def show_data_main():
    np.random.seed(2)
    x = np.concatenate([np.random.normal(0, 2, 2000),
                        np.random.normal(5, 5, 2000),
                        np.random.normal(3, 0.5, 600)])
    plt.hist(x, 80, density=True)
    plt.xlim(-10, 20)
    plt.show()


def gmm_main():
    np.random.seed(2)
    # x = np.concatenate([np.random.normal(0, 2, 2000),
    #                     np.random.normal(5, 5, 2000),
    #                     np.random.normal(3, 0.5, 600)])
    shifted_gaussian = np.random.randn(300, 2) + np.array([20, 20])
    C = np.array([[0., -0.7], [3.5, .7]])
    stretched_gaussian = np.dot(np.random.randn(300, 2), C)
    X_train = np.vstack([shifted_gaussian, stretched_gaussian])

    clf = GaussianMixture(n_components=4, max_iter=500, random_state=3).fit(X_train)

gmm_main()