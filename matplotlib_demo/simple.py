# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt


def sin_main():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    plt.plot(X, C)
    plt.plot(X, S)

    plt.figure(figsize=(8, 6), dpi=80)
    plt.subplot(1, 1, 1)
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C, S = np.cos(X), np.sin(X)
    # 绘制余弦曲线，使用蓝色的、连续的、宽度为 1 （像素）的线条
    plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")

    # 绘制正弦曲线，使用绿色的、连续的、宽度为 1 （像素）的线条
    plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")
    plt.xlim(-4.0, 4.0)

    # 设置横轴记号
    plt.xticks(np.linspace(-4, 4, 9, endpoint=True))

    # 设置纵轴的上下限
    plt.ylim(-1.0, 1.0)

    # 设置纵轴记号
    plt.yticks(np.linspace(-1, 1, 5, endpoint=True))

    # 以分辨率 72 来保存图片
    # savefig("exercice_2.png",dpi=72)

    plt.show()


sin_main()
