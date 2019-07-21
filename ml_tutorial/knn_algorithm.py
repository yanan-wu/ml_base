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
    k_nearest_neighbors(dataset, new_features, 2)
    # [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
    # plt.show()


def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('k is set to a value less than total voting grops')

    distances = []
    for group in data:
        for features in data[group]:
            euc_distance = np.linalg.norm(np.array(features) - np.array(predict))
            distances.append([euc_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    return vote_result


prepare_dataset()
