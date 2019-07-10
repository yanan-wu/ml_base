# coding: utf-8

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt


def show_main():
    xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
    ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)
    m, b = best_fit_slope_and_intertcept(xs, ys)

    regression_line = [(m * x) + b for x in xs]
    r_squared = coefficient_of_determination(ys, regression_line)
    print(r_squared)
    plt.scatter(xs, ys)
    plt.plot(xs, regression_line)
    plt.show()


def best_fit_slope_and_intertcept(xs, ys):
    m = ((mean(xs) * mean(ys) - mean(xs * ys)) / (mean(xs) * mean(xs) - mean(xs * xs)))
    b = mean(ys) - m * mean(xs)
    return m, b


def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)


def squared_error(ys_orig, ys_line):
    return sum((ys_orig - ys_line)**2)


show_main()
