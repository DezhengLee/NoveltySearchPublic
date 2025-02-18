import numpy as np
from scipy.stats import levy_stable
from random import gauss
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from sympy.stats.sampling.sample_scipy import scipy


def finLevyFlight(alpha, beta, scale, dim, dirList, step):
    # Assume starts at original
    # Number of alternative direction dirList should be greater than the length of step
    start = np.zeros(dim)
    res = np.zeros([step, dim])
    res[0] = start
    for _ in range(1, step):
        res[_] = res[_ - 1] + dirList[_] * scale * levy_stable.rvs(alpha, beta)
    return res


def genNDimDir(num, dim):
    dir_list = np.zeros([num, dim])
    for _ in range(num):
        dir_list[_] = make_rand_vector(dim)
    return dir_list


def make_rand_vector(dims):
    # function source: https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
    # Based on the fact that if X = (X_1, ..., X_n) iid ~ N(0, 1), then X/sqrt(sum_i{X_i^2}) is ...
    # uniformly distributed on the surface of unit sphere.
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def cos_basis(num_term, x):
    xleng = len(x)
    res = np.zeros([xleng, num_term])
    for k in range(0, num_term):
        res[:, k] = [np.cos(k * np.pi * i) for i in x]
    return res

def cosFourier(x: np.array, coef: np.array) ->  np.array:
    num_term = len(coef)
    res = np.dot(cos_basis(num_term, x), coef)
    return res


if __name__ == '__main__':
    num_steps = 20000
    alpha = 1.2
    beta = 0.5
    scale = 0.001
    dim = 20
    t_step = 1000
    t = np.linspace(0, 1, t_step)

    test_coef = finLevyFlight(alpha, beta, scale, dim, genNDimDir(2, dim), 2)

    plt.plot(t, cosFourier(t, test_coef[1,:]))
    plt.show()

    # d2list = genNDimDir(num_steps, dim=dim)
    # flight = finLevyFlight(alpha, beta, scale, dim, d2list, num_steps)
    # # plt.plot(flight[:, 0], flight[:, 1],flight[:, 2],  linestyle="-", marker="o", markersize=2, alpha=0.7)
    # # plt.scatter(flight[0][0], flight[0][1], flight[0][2], color="red", marker="o", label="Start")
    # # plt.scatter(flight[-1][0], flight[-1][1],flight[-1][2], color="blue", marker="x", label="End")
    # # plt.title(r"Lévy Flight $\alpha={}, \beta={}, steps={}$".format(alpha, beta, num_steps))
    # # plt.xlabel("X Position")
    # # plt.ylabel("Y Position")
    # # # plt.zlabel('Z Position')
    # # plt.legend()
    # # plt.grid()
    # # plt.show()
    #
    #
    #
    # # x, y = d2LevyFlight(alpha, beta, scale, size)
    # #
    # # plt.figure(figsize=(8, 6))
    # # plt.plot(x, y, linestyle="-", marker="o", markersize=2, alpha=0.7)
    # # plt.scatter(x[0], y[0], color="red", marker="o", label="Start")
    # # plt.scatter(x[-1], y[-1], color="blue", marker="x", label="End")
    # # plt.title(r"Lévy Flight $\alpha={}, \beta={}, steps={}$".format(alpha, beta, num_steps))
    # # plt.xlabel("X Position")
    # # plt.ylabel("Y Position")
    # # plt.legend()
    # # plt.grid()
    # # plt.show()
    #
    #
    # # Create the 3D figure
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the 3D path
    # ax.plot(flight[:, 0], flight[:, 1],flight[:, 2], label="3D Path (Helix)", color='b')
    #
    # # Labels
    # ax.set_xlabel("X-axis")
    # ax.set_ylabel("Y-axis")
    # ax.set_zlabel("Z-axis")
    # ax.set_title("3D Path Plot")
    #
    # # Show the plot
    # plt.show()
