import numpy as np
from DSvisualizer import DSvisualizer

Etol = 1e-5

def hehe_ymax(e):
    # viete-cardano
    a, b, c, d = 1./3., -0.5, 0., float(e)
    p = (3.*a*c - b**2.)/(3.*a**2.)
    q = (2.*b**3. - 9.*a*b*c + 27.*a**2. *d)/(27.*a**3.)
    k = 1.
    tk = 2.*np.power(-p/3., 0.5, dtype=complex)*np.cos((1./3.)*np.arccos((3.*q*0.5/p)*np.power(-3./p, 0.5, dtype=complex))-2.*np.pi*k/3.)
    y_max = tk-b/(3.*a)
    return y_max

def hehe(t, y, par=None):
    # Henon-Heiles galactic potential:
    # y is of the form [x, vx, y, vy]
    v = y[1::2]
    ax1 = - y[0] - 2.*y[0]*y[2]
    ay1 = - y[2] - np.power(y[0],2.) + np.power(y[2],2.)
    return [v[0], ax1, v[1], ay1]

def heheEnergy(y):
    return 0.5*(y[1]**2. + y[3]**2.) + 0.5*(y[0]**2. + y[2]**2.) + y[0]**2. *y[2] - (1./3.)*y[2]**3.

def heheYsample(e, nTracks):
    y0_ar = []
    yc_ar = np.linspace(0.1, 0.9, int(nTracks))*hehe_ymax(e)
    for i in range(nTracks):
        y = np.zeros(4)
        y[2] = yc_ar[i]
        p = np.sqrt(2.*(e - 0.5*y[2]**2. + (1./3.)*y[2]**3.))
        theta = np.random.uniform(np.pi*0.5, 3.*np.pi*0.5)

        y[1] = p*np.cos(theta)
        y[3] = p*np.sin(theta)
        dE = heheEnergy(y) - e
        if dE > Etol: 
            print(y, heheEnergy(y), dE)
            raise Exception("Sampling Energy doesn't match!")
        y0_ar.append(y)
    return np.array(y0_ar)

def hehePick(e, xy_tuple):
    y = np.zeros(4)
    y[2] = xy_tuple[0]
    y[3] = xy_tuple[1]
    de = e-heheEnergy(y)
    if de >= 0.:
        y[1] = np.sqrt(2.*de)
        return y
    else: return None

DSvisualizer(hehe, heheYsample, heheEnergy, hehePick, 100, 1./6., 200, (1./100., 1./6.), (100, 1000), (2, 3), (0, 2))