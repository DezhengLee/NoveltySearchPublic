import numpy as np
from dask.order import order
import matplotlib.pyplot as plt
import LevyFlight
import FunctionalBasis as fb
import FunctionCombiner as fc
from Hmap import *



if __name__ == '__main__':
    num_steps = 30000
    alpha = 1
    beta = 0.5
    scale = 0.001
    dim = 2
    seed = 30

    xlist = np.linspace(-1, 1, 1000)

    t_step = 1000
    t = np.linspace(0, 1, t_step)

    testPoint = 20

    flight = LevyFlight.LevyFlight(alpha=alpha, beta=beta, scale=scale, dim=dim, seed=seed)
    flightPath = flight.gen_flight(steps=num_steps)
    coefTest = flightPath[testPoint]
    fbasis = fb.FourierBasis(order=dim)

    func = fc.FunctionCombiner(basis=fbasis, coefficients=flightPath)

    fpoints = func.evaluate(xlist)

    plt.plot(xlist, fpoints[testPoint])
    plt.show()

    alphaH = 0.4
    zsteps = 100
    zlist = np.linspace(0, 1, zsteps)
    Hres = flight.heatmapData()

    plt.figure(figsize=(8, 6))
    heatmap(Hres)
    plt.show()