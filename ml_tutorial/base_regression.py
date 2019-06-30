# coding: utf-8

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


def show_main():
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)
    m = best_fit_slope(xs, ys)
    print(m)


def best_fit_slope(xs, ys):
    m = ((mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) * mean(xs) - mean(xs * xs)))
    return m

show_main()
