import numpy as np
from functools import reduce
from matplotlib import pyplot as plt
from scipy.sparse import kron, csr_matrix

from .cad import point_on_bspline_curve
from .cad import point_on_bspline_surface
from .bsplines import hrefinement_matrix


__all__ = ['plot_field_1d', 'plot_field_2d', 'prolongation_matrix']

# ==========================================================
def plot_field_1d(knots, degree, u, nx=101, color='b', xmin = None, xmax = None, label = None):
    n = len(knots) - degree - 1

    if xmin is None :
        xmin = knots[degree]
    if xmax is None :
        xmax = knots[-degree-1]

    xs = np.linspace(xmin, xmax, nx)

    P = np.zeros((len(u), 1))
    P[:,0] = u[:]
    Q = np.zeros((nx, 1))
    for i,x in enumerate(xs):
        Q[i,:] = point_on_bspline_curve(knots, P, x)

    if label is not None :
        plt.plot(xs, Q[:,0], label = label)
    else :
        plt.plot(xs, Q[:,0])

# ==========================================================
def plot_field_2d(knots, degrees, u, nx=101, ny=101):
    T1,T2 = knots
    p1,p2 = degrees

    n1 = len(T1) - p1 - 1
    n2 = len(T2) - p2 - 1

    xmin = T1[p1]
    xmax = T1[-p1-1]

    ymin = T2[p2]
    ymax = T2[-p2-1]

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    n1,n2 = u.shape

    P = np.zeros((n1, n2, 1))
    P[:,:,0] = u[:,:]
    Q = np.zeros((nx, ny, 1))
    for i1,x in enumerate(xs):
        for i2,y in enumerate(ys):
            Q[i1,i2,:] = point_on_bspline_surface(T1, T2, P, x, y)
    X,Y = np.meshgrid(xs,ys)
    plt.contourf(X, Y, Q[:,:,0])

# ==========================================================
def prolongation_matrix(VH, Vh):
    # TODO not working for duplicated internal knots

    # ... TODO check that VH is included in Vh
    # ...

    # ...
    mats = []
    for Wh, WH in zip(Vh.spaces, VH.spaces):
        ths = Wh.knots
        tHs = WH.knots
        ts = set(ths) - set(tHs)
        ts = np.array(list(ts))

        M = hrefinement_matrix( ts, Wh.degree, tHs )
        mats.append(csr_matrix(M))
    # ...

    M = reduce(kron, (m for m in mats))

    return M
