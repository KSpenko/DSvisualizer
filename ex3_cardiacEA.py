import numpy as np
from DSvisualizer import DSvisualizer

MAX = 1.

def cardiacEA(t, y, par=None):
    # Siroos Nazari
    # https://www.researchgate.net/publication/274122953_Modified_Modeling_of_the_Heart_by_Applying_Nonlinear_Oscillators_and_Designing_Proper_Control_Signal
    d1, d2, d3, c1, c2, c3, r35, r13, r15, r31, r51, r53, a1, a2, a3, w = par
    # SA
    dy1 = y[1]
    dy2 = -d1*(y[0]**2.-1.)*y[1] - c1*y[0] + a1*np.cos(w*t) + r13*(y[0]-y[2]) + r15*(y[0]-y[4])
    # AV
    dy3 = y[3]
    dy4 = -d2*(y[2]**2.-1.)*y[3] - c2*y[2] + a2*np.cos(w*t) + r31*(y[2]-y[0]) + r35*(y[2]-y[4])
    # HP
    dy5 = y[5]
    dy6 = -d3*(y[4]**2.-1.)*y[5] - c3*y[4] + a3*np.cos(w*t) + r51*(y[4]-y[0]) + r53*(y[4]-y[2])
    return [dy1, dy2, dy3, dy4, dy5, dy6]

def cardioYsample(nTracks, par=None):
    return np.random.uniform(-MAX, MAX, (nTracks, 6))

def cardioPick(xy_tuple, par=None):
    y = np.random.uniform(-MAX, MAX, 6)
    y[0] = 0.
    y[2] = xy_tuple[0]
    y[4] = xy_tuple[1]
    return y

def cardioPmap(yp):
    return [yp[0], yp[2]]

def cardioTraj(y, t=None):
    return t, np.sum([y[0], y[2], y[4]], axis=0)

def cardio3D(y):
    return [y[0],y[2],y[4]]

def cardioFFT(y):
    return np.sum([y[0], y[2], y[4]], axis=0)

cardio_labels = {
    "poincare": (r'$SA$',r'$AV$'),
    "trajectory": (r'$t$', r'$SA+AV+HP$'),
    "3D": (r'$SA$', r'$AV$', r'$HP$'),
    "FFT": r'$(SA+AV+HP)$',
}

cardioPar = {
    r'$d_1$': [5., (-5., 10.)],
    r'$d_2$': [6., (-5., 10.)],
    r'$d_3$': [7., (-5., 10.)],
    r'$c_1$': [1.7, (0., 2.)],
    r'$c_2$': [0.5, (0., 2.)],
    r'$c_3$': [1., (0., 2.)],
    r'$r_{35}$': [1., (0., 1.)],
    r'$r_{13}$': [1., (0., 1.)],
    r'$r_{15}$': [1., (0., 1.)],
    r'$r_{31}$': [0.0001, (0., 1.)],
    r'$r_{51}$': [0.0001, (0., 1.)],
    r'$r_{53}$': [0.0001, (0., 1.)],
    r'$a_1$': [5., (-5., 10.)],
    r'$a_2$': [5., (-5., 10.)],
    r'$a_3$': [5., (-5., 10.)],
    r'$\omega$': [5., (-5., 10.)],
}

DSvisualizer(cardiacEA, cardioYsample, cardioPick, 100, cardioPmap, cardioTraj, cardio3D, cardioFFT, (100, 10, 500), cardioPar, cardio_labels, poincInd=4, maxstep=1., equal=False)