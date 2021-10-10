import numpy as np
from DSvisualizer import DSvisualizer

Etol = 1e-1
RMAX = 10.
QMAX = np.sqrt(RMAX)
time = 200
Nstep = 1
max_step = 0.01

def KStransform(r):
    q1 = np.sqrt(r[0])
    q2 = np.sqrt(r[1])
    p1 = 2.*r[2]*q1
    p2 = 2.*r[3]*q2
    return np.array([q1, q2, p1, p2])

def KSinverse(q):
    r1 = np.power(q[0], 2)
    r2 = np.power(q[1], 2)
    p1 = 0.5*q[2]/q[0]
    p2 = 0.5*q[3]/q[1]
    return np.array([r1, r2, p1, p2])

def colHeEnergy(y, par=None):
    E, Z = par
    yr = KSinverse(y)
    ene = 0.
    ene += 0.5*(yr[2]**2. + yr[3]**2.)
    ene -= Z*((1./yr[0])+(1./yr[0]))
    ene += 1./(yr[0]+yr[1])
    return ene

def colHeYsample(nTracks, par=None):
    e, Z = par
    y0_ar = []
    q2_ar = np.linspace(0.1, 0.9, int(nTracks))*np.sqrt(-7./(2.*e))
    for i in range(nTracks):
        q2 = q2_ar[i]
        Qhigh_max = np.sqrt((-(q2**2.)*(3.+(q2**2.)*e)-np.sqrt((q2**4.)*((3.+(q2**2.)*e)**2.-8.*(2.+(q2**2.)*e)) ) )/(2.*(2.+(q2**2.)*e)))
        Qhigh_max = np.nanmin([Qhigh_max, QMAX])
        q1 = np.random.choice([-1,1])*np.random.uniform(Qhigh_max*0.1, Qhigh_max)

        r12 = np.sqrt(q1**2.+q2**2.)
        h_neg = -2.*r12**2. + q1**2.*q2**2.*(1/(r12**2.) - e)
        ratio = np.random.uniform(0., 1.)
        p1 = np.random.choice([-1,1])*np.sqrt(8.*ratio*(-h_neg)/(q2**2.))
        p2 = np.random.choice([-1,1])*np.sqrt(8.*(1-ratio)*(-h_neg)/(q1**2.))
        h = (1./8.)*(q2**2.*p1**2.+q1**2.*p2**2.) + h_neg
        if h > Etol:
            raise Exception("H'(E) = "+str(h))
        y0_ar.append( np.array([q1, q2, p1, p2]) )
    return y0_ar

def colHe(t, y, par=None):
    E, Z = par
    y0_2 = y[0]*y[0]
    y1_2 = y[1]*y[1]
    dq1 = 0.25*y[2]*y1_2
    dq2 = 0.25*y[3]*y0_2
    q1q2 = y[0]/y[1]
    dp1 = y[0]* (4. - 0.25*y[3]*y[3] + 2.*y1_2*E - 2./( q1q2**4. + 2.*(q1q2**2.) + 1.) )
    q2q1 = y[1]/y[0]
    dp2 = y[1]* (4. - 0.25*y[2]*y[2] + 2.*y0_2*E - 2./( q2q1**4. + 2.*(q2q1**2.) + 1.) )
    return np.array([dq1, dq2, dp1, dp2])

def colHePick(xy_tuple, par=None):
    E, Z = par
    y = np.zeros(4)
    y[1] = xy_tuple[0]
    y[2] = 4.
    y[3] = xy_tuple[1]*2.*y[1]
    de = E-colHeEnergy(y, par=par)
    if np.abs(de) >= Etol: return None
    else: return y

def colHePmap(yp):
    return [np.abs(yp[1]), 0.5*yp[3]/yp[1]]

def colHeTraj(y, t=None):
    return np.power(y,2)[:2]

def colHe3d(y):
    return [np.power(y[0],2), np.power(y[1],2), 0.5*y[3]/y[1]]

def colHeFFT(y):
    return np.power(y,2)[:2]

colHe_labels = {
    "poincare": (r'$\sqrt{r_2}$',r'$p_2$'),
    "trajectory": (r'$r_1$', r'$r_2$'),
    "3D": (r'$r_1$', r'$r_2$', r'$p_2$'),
    "FFT": r'$(r_1^2+r_2^2)$',
}

colHePar = {
    "Energy": [-0.5, (-10, 0.)],
    "Z": [2, (1, 10)],
}

DSvisualizer(colHe, colHeYsample, colHePick, 10, colHePmap, colHeTraj, colHe3d, colHeFFT, (20, 10, 500), colHePar, colHe_labels, maxstep=0.1)