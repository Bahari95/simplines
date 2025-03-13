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


from simplines.linalg import StencilVector
from simplines.results_f90 import least_square_Bspline, pyccel_sol_field_2d
from numpy import exp
from numpy import cos
from numpy import sin
from numpy import pi
from numpy import arctan2
from numpy import sqrt
from numpy import cosh
from numpy import zeros
from numpy import empty
def build_dirichlet(V, f, map = None, admap = None):
    '''
    V    : FE space
    f[0] : on the left
    f[1] : on the right
    f[2] : on the bottom
    f[3] : on the top
    map = (x,y) : control points
    admap = (x, V1, y, V2) control points and associated space
    '''
    if len(f) > 1 :
        fx0      = lambda x,y :  eval(f[0])
        fx1      = lambda x,y :  eval(f[1])
        fy0      = lambda x,y :  eval(f[2])
        fy1      = lambda x,y :  eval(f[3])
    elif map is None:
        xi       = V.grid[0]
        yi       = V.grid[1]
        sol      = lambda x,y :  eval(f[0]) 
        fx0      = lambda   y :  sol(xi[0],y)
        fx1      = lambda   y :  sol(xi[-1],y) 
        fy0      = lambda x   :  sol(x,yi[0])
        fy1      = lambda x   :  sol(x,yi[-1])
    else :
        fx0      = lambda x,y :  eval(f[0])
        fx1      = lambda x,y :  eval(f[0])
        fy0      = lambda x,y :  eval(f[0])
        fy1      = lambda x,y :  eval(f[0]) 
    u_d          = StencilVector(V.vector_space)
    x_d          = np.zeros(V.nbasis)
    if V.dim == 2:
        if map is None:
            #------------------------------
            #.. In the parametric domain
            x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0)
            x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1)
            x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0)
            x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1)

        elif admap is None :
            #------------------------------
            #.. In the phyisacl domain without adaptive mapping               
            n_dir        = V.nbasis[0] + V.nbasis[1]+100
            sX           = pyccel_sol_field_2d((n_dir,n_dir),  map[0] , V.knots, V.degree)[0]
            sY           = pyccel_sol_field_2d((n_dir,n_dir),  map[1] , V.knots, V.degree)[0]

            x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0(sX[0, :], sY[ 0,:]), m= n_dir)
            x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1(sX[-1,:], sY[-1,:]), m= n_dir)
            x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0(sX[:, 0], sY[:, 0]), m= n_dir)
            x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1(sX[:,-1], sY[:,-1]), m= n_dir)

        else :
            #------------------------------
            #.. In the phyisacl domain with adaptive mapping               
            n_dir        = V.nbasis[0] + V.nbasis[1]+100

            Xmae         = pyccel_sol_field_2d((n_dir,n_dir),  admap[0] , admap[2].knots, admap[2].degree)[0]
            Ymae         = pyccel_sol_field_2d((n_dir,n_dir),  admap[1] , admap[3].knots, admap[3].degree)[0]
            sX           = pyccel_sol_field_2d( None, map[0], V.knots, V.degree, meshes = (Xmae, Ymae))[0]
            sY           = pyccel_sol_field_2d( None, map[1], V.knots, V.degree, meshes = (Xmae, Ymae))[0]

            x_d[ 0 , : ] = least_square_Bspline(V.degree[1], V.knots[1], fx0(sX[0, :], sY[ 0,:]), m= n_dir)
            x_d[ -1, : ] = least_square_Bspline(V.degree[1], V.knots[1], fx1(sX[-1,:], sY[-1,:]), m= n_dir)
            x_d[ : , 0 ] = least_square_Bspline(V.degree[0], V.knots[0], fy0(sX[:, 0], sY[:, 0]), m= n_dir)
            x_d[ : ,-1 ] = least_square_Bspline(V.degree[0], V.knots[0], fy1(sX[:,-1], sY[:,-1]), m= n_dir)
    if V.dim == 3 :
        print("not yet implemented")
        #.. TODO : USE l2 PROJECTION USING FAST DIAG
    u_d.from_array(V, x_d)
    return x_d, u_d