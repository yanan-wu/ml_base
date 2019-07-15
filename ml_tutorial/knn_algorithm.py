# @Time: 2019/7/16 7:19
# @Auth: yanan.wu
# @Desc:

import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
import warnings
from collections import Counter


def prepare_dataset():
    style.use('fivethirtyeight')
    dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
    new_features = [5, 7]
    [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
    plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting grops')
    return None



prepare_dataset()
