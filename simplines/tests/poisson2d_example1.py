"""
poisson2d_example1.py

A basic test for periodic boundary conditions.(2D)

author :  M. BAHARI
"""


from simplines import compile_kernel, apply_dirichlet, apply_periodic

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

#.. Prologation by knots insertion matrix
from simplines import prolongation_matrix


#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
#%matplotlib inline
import timeit
import time
start = time.time()

from gallery_section_00 import assemble_stiffnessmatrix1D
from gallery_section_00 import assemble_massmatrix1D

assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

#---In Poisson equation
from gallery_section_00 import assemble_vector_ex01 #---1 : In uniform mesh
from gallery_section_00 import assemble_norm_ex01 #---1 : In uniform mesh

assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2  = compile_kernel(assemble_norm_ex01, arity=1)

print('time to import utilities of Poisson equation =', time.time()-start)


from scipy.sparse import kron
from scipy.sparse import csr_matrix
from simplines import Poisson
from scipy.sparse import csc_matrix, linalg as sla
from numpy import zeros, linalg, asarray
import numpy as np

from tabulate import tabulate

#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists


#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2 , V, u_p, xuh, periodic):

       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition
       u  = StencilVector(V.vector_space)
       
       #..Stiffness and Mass matrix in 1D in the first deriction
       K1 = assemble_stiffness1D(V1)
       K1 = apply_periodic(V1, K1)
       K1 = csr_matrix(K1)

       M1 = assemble_mass1D(V1)
       M1 = apply_periodic(V1, M1)
       M1 = csr_matrix(M1)

       # Stiffness and Mass matrix in 1D in the second deriction
       K2 = assemble_stiffness1D(V2)
       K2 = K2.tosparse()
       K2 = K2.toarray()[1:-1,1:-1]
       K2 = csr_matrix(K2)

       M2 = assemble_mass1D(V2)
       M2 = M2.tosparse()
       M2 = M2.toarray()[1:-1,1:-1]
       M2 = csr_matrix(M2)

       mats_1 = [M1, K1]
       mats_2 = [M2, K2]

       # ...
       poisson = Poisson(mats_1, mats_2)
      
       #---- assemble rhs
       rhs              = assemble_rhs( V, fields = [u_p])
       # ...
       b                = apply_periodic(V, rhs, periodic)
       b                = b.reshape(V1.nbasis-V1.degree,V2.nbasis)
       b                = b[:, 1:-1]       
       b                = b.reshape((V1.nbasis-V1.degree)*(V2.nbasis-2))
       #--Solve a linear system
       xk               = poisson.solve(b)
       xk               = xk.reshape((V1.nbasis-V1.degree, V2.nbasis-2))
       x                = zeros((V1.nbasis-V1.degree,V2.nbasis))
       x[:,1:-1]        = xk[:,:]
       x[:,:]          += xuh[:-V1.degree,:]
       x                = apply_periodic(V, x, periodic, update = True)       
       u.from_array(V, x)

       Norm    = assemble_norm_l2(V, fields=[u])
       norm    = Norm.toarray()
       l2_norm = norm[0]
       H1_norm = norm[1]
       
       return u, x, l2_norm, H1_norm

degree     = 3
nelements  = 64

#---------------------------------------
from numpy import sin, exp, pi, linspace
u_exact = lambda x, y : x**2*sin(5.*pi*y) #+ 1.0*exp(-((x-0.5)**2 + (y-0.5)**2)/0.02)

#fx0 = lambda x : u_exact(-pi,x) #if x<0.25+100.*t else 1.
#fx1 = lambda x : u_exact(pi,x)
fy0 = lambda x : u_exact(x,-1.) #if x<0.25+100.*t else 1.
fy1 = lambda x : u_exact(x,1.)

#-------------------------------------------
# create the spline space for each direction
grids     = linspace(-0.5*pi, 0.5*pi,  nelements + 1)
V1 = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2, periodic=True)
#... We use refinded mesh in the position where we have a Gaussian
grids     = linspace(-1., 1.,  nelements + 1)
V2        = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2)

# create the tensor space
Vh = TensorSpace(V1, V2)

# compute the interpolate B-spline function for the Dirichlet boundary condition using least square method
from simplines import least_square_Bspline

# ... Dirichlet boundary condition 
x0uh                         = zeros(Vh.nbasis)
#x0uh[0, : ]                  = least_square_Bspline(degree, V2.knots, fx0)
#x0uh[V1.nbasis-1, : ]        = least_square_Bspline(degree, V2.knots, fx1)
x0uh[: ,0]                   = least_square_Bspline(degree, V1.knots, fy0)
x0uh[: ,V2.nbasis-1]         = least_square_Bspline(degree, V1.knots, fy1)
u_p                          = StencilVector(Vh.vector_space)
u_p.from_array(Vh, x0uh)

print('#---IN-UNIFORM--MESH')
u_ph, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, Vh, u_p, x0uh, [True, False])
print('l2_norm = {} H1_norm = {} '.format(l2_norm, H1_norm) ) 
nbpts    = 80

print(xuh.shape)
u_poisson, u_xpoisson, u_ypoisson, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  xuh , Vh.knots, Vh.degree)

#print(np.max(u_xpoisson), np.min(u_xpoisson) ) 


figtitle  = 'solution of poisson equation in the new geometry'
fig, axes       = plt.subplots( 1, 2, figsize=[12,12], gridspec_kw={'width_ratios': [2, 2]} , num=figtitle )
for ax in axes:
   ax.set_aspect('equal')
axes[0].set_title( ' Approximate Solution' )
ima = axes[0].contourf( X, Y, u_poisson, cmap= 'jet')
divider = make_axes_locatable(axes[0]) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(ima, cax=cax)

axes[1].set_title( ' Exact Solution' )
im = axes[1].contourf( X, Y, u_exact(X, Y), cmap= 'jet')
divider = make_axes_locatable(axes[1]) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im, cax=cax)
fig.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.savefig('figs/periodic_Poisson')
plt.show()