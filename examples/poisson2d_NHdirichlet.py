"""
poisson2d_NHdirichlet.py

author :  M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet
from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

#.. Prologation by knots insertion matrix
from simplines import least_square_Bspline


#from matplotlib.pyplot import plot, show
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import LinearLocator, FormatStrFormatter
#%matplotlib inline
import timeit
import time
start = time.time()

#---In Poisson equation
from examples.gallery.gallery_section_04 import assemble_vector_ex01 #---1 : In uniform mesh
from examples.gallery.gallery_section_04 import assemble_matrix_ex01 #---1 : In uniform mesh
from examples.gallery.gallery_section_04 import assemble_norm_ex01 #---1 : In uniform mesh

assemble_stiffness2D = compile_kernel(assemble_matrix_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)
print('time to import utilities of Poisson equation =', time.time()-start)

from scipy.sparse        import csr_matrix
from scipy.sparse        import csc_matrix, linalg as sla
from scipy.sparse.linalg import gmres
from numpy               import zeros, linalg, asarray
from numpy               import cos, pi
import numpy as np

from tabulate import tabulate
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists
#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2 , V, x_d = None, u_d = None):

       u   = StencilVector(V.vector_space)

       stiffness           = assemble_stiffness2D(V)
       stiffness     = (stiffness.tosparse()).toarray()
       stiffness     = stiffness.reshape((V1.nbasis,V2.nbasis,V1.nbasis,V2.nbasis))
       stiffness     = stiffness[1:-1,1:-1,1:-1,1:-1]
       stiffness     = stiffness.reshape(((V1.nbasis-2)*(V2.nbasis-2),(V1.nbasis-2)*(V2.nbasis-2)))
       #stiffness           = apply_dirichlet(V, stiffness)
       #--Assembles matrix
       #stiffness             = stiffness.tosparse()
       lu                  = sla.splu(csc_matrix(stiffness))
       #--Assembles right hand side of Poisson equation
       rhs                 = assemble_rhs( V, fields = [u_d] )
       rhs            = rhs.toarray()
       rhs            = rhs.reshape((V1.nbasis,V2.nbasis))
       rhs            = rhs[1:-1,1:-1]
       b              = rhs.reshape((V1.nbasis-2)*(V2.nbasis-2))
       # rhs                 = apply_dirichlet(V, rhs)
       #b                   = rhs.toarray()
       # ...
       x                   = lu.solve(b)       
       x                   = x.reshape((V1.nbasis-2, V2.nbasis-2))       

       # Rassembles Direcjlet boundary conditions
       xsol                = zeros(V.nbasis)
       xsol[ 1: -1, 1: -1] = x[ : , : ]
       xsol[ : , : ]      += x_d[ : , : ]       
       u.from_array(V, xsol)
       #--Computes error l2 and H1
       Norm                = assemble_norm_l2(V, fields=[u])
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             =  norm[1]
       print('<.> l2_norm = {}  ||u||_H1= {} using nelement={} degree={} '.format(l2_norm, H1_norm, V.nelements, V.degree))
       return u, xsol, l2_norm, H1_norm


degree     = 4
#nelements = 64
table = zeros((5,5))
i = 1

#..... Initialisation and computing optimal mapping for 16*16

nelements  = 64

#----------------------
# create the spline space for each direction
V1 = SplineSpace(degree=degree, nelements=nelements, nderiv = 2)
V2 = SplineSpace(degree=degree, nelements=nelements, nderiv = 2)

# create the tensor space
Vh = TensorSpace(V1, V2)

#..
fx0 = lambda x :  0.
fx1 = lambda x :  2.* cos(pi*x)
fy0 = lambda x :  2.* x 
fy1 = lambda x : -2.* x 

#------------------------------
u_d                  = StencilVector(Vh.vector_space)
x_d                  = np.zeros(Vh.nbasis)
x_d[0, : ]           = least_square_Bspline(degree, V2.knots, fx0)
x_d[V1.nbasis-1, : ] = least_square_Bspline(degree, V2.knots, fx1)
x_d[:,0]             = least_square_Bspline(degree, V1.knots, fy0)
x_d[:, V2.nbasis-1]  = least_square_Bspline(degree, V1.knots, fy1)

u_d.from_array(Vh, x_d)

print('#---IN-UNIFORM--MESH')
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, Vh, x_d = x_d, u_d = u_d)
xuh_uni = xuh

#---Compute a solution
nbpts = 50
#---Solution in uniform mesh
u, ux, uy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  xuh , Vh.knots, Vh.degree)

#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface

# set up a figure twice as wide as it is tall
fig = plt.figure(figsize=plt.figaspect(0.5))
#===============
#  First subplot
# set up the axes for the first plot
ax = fig.add_subplot(1, 2, 1, projection='3d')

# plot a 3D surface like in the example mplot3d/surface3d_demo
surf0 = ax.plot_surface(X, Y, u, rstride=1, cstride=1, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Approximate solution')
ax.set_xlabel('X',  fontweight ='bold')
ax.set_ylabel('Y',  fontweight ='bold')
# Add a color bar which maps values to colors.
fig.colorbar(surf0, shrink=0.5, aspect=25)

#===============
# Second subplot
Sol = lambda x, y : 2.*x*cos(pi*y)
ax = fig.add_subplot(1, 2, 2, projection='3d')
surf = ax.plot_surface(X, Y, Sol(X,Y), cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
ax.set_xlim(0.0, 1.0)
ax.set_ylim(0.0, 1.0)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_title('Exact Solution')
ax.set_xlabel('F1',  fontweight ='bold')
ax.set_ylabel('F2',  fontweight ='bold')
fig.colorbar(surf, shrink=0.5, aspect=25)
plt.savefig('figs/Poisson3D.png')
plt.show()
