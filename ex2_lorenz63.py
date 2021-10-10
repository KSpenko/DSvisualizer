import numpy as np
from DSvisualizer import DSvisualizer

MAX = 40.

def lorenzYsample(nTracks, par=None):
    return np.random.uniform(-MAX, MAX, (nTracks, 3))

def lorenz(t, y, par=None):
    sigma, rho, beta = par
    dx = sigma*(y[1] - y[0])
    dy = y[0]*(rho - y[2]) - y[1]
    dz = y[0]*y[1] - beta*y[2]
    return np.array([dx, dy, dz])

def lorenzPick(xy_tuple, par=None):
    y = np.zeros(3)
    y[1] = xy_tuple[0]
    y[2] = xy_tuple[1]
    return y

def defaultPmap(yp):
    return [yp[1], yp[2]]

def defaultTraj(y, t=None):
    return y[:2]

def default3D(y):
    return y

def defaultFFT(y):
    return np.sum(np.power(y,2), axis=0)

lorenzPar = {
    r'$\sigma$': [10., (0., 100.)],
    r'$\rho$': [28., (0., 100.)],
    r'$\beta$': [8./3., (0., 10.)],
}

lorenz_labels = {
    "poincare": (r'$y$',r'$z$'),
    "trajectory": (r'$x$', r'$y$'),
    "3D": (r'$x$', r'$y$', r'$z$'),
    "FFT": r'$(x^2+y^2+z^2)$',
}

DSvisualizer(lorenz, lorenzYsample, lorenzPick, 100, defaultPmap, defaultTraj, default3D, defaultFFT, (20, 10, 100), lorenzPar, lorenz_labels, maxstep=0.1)