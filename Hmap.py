import numpy
import numpy as np
from plotly.express import imshow
from seaborn import heatmap

from main import *


def H(x: numpy.ndarray, z: numpy.ndarray, alpha: float, n: int) -> float:
    if np.any((z < 0) | (z > 1)):
        raise ValueError('z must be between 0 and 1')
    k_vals = np.arange(1, n + 1)
    coeffs = np.sqrt(2) * k_vals ** (-alpha)

    return np.einsum('jk,zk->zj', x[:, :n], coeffs * np.cos(np.outer(z, k_vals) * np.pi))




if __name__ == '__main__':
    num_steps = 20000
    alpha = 1.2
    beta = 0.7
    scale = 0.001
    dim = 4
    t_step = 1000

    alphaH = 0.4

    zsteps = 100


    dirList = genNDimDir(num_steps, dim=dim)

    flight = finLevyFlight(alpha, beta, scale, dim, dirList, num_steps)

    zlist = np.linspace(0, 1, zsteps)
    Hres = H(flight, zlist, alphaH, dim)

    # plt.plot(flight[:, 0], flight[:, 1], linestyle="-", marker="o", markersize=2, alpha=0.7)
    # plt.scatter(flight[0][0], flight[0][1], color="red", marker="o", label="Start")
    # plt.scatter(flight[-1][0], flight[-1][1], color="blue", marker="x", label="End")
    # plt.title(r"Lévy Flight $\alpha={}, \beta={}, steps={}$".format(alpha, beta, num_steps))
    # plt.xlabel("X Position")
    # plt.ylabel("Y Position")
    # plt.legend()
    # plt.grid()
    # plt.show()

    plt.figure(figsize=(8, 6))
    heatmap(Hres)
    plt.show()




