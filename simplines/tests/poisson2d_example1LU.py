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

#---In Poisson equation
from gallery_section_00 import assemble_vector_ex01    #---1 : In uniform mesh
from gallery_section_00 import assemble_norm_ex01      #---1 : In uniform mesh
from gallery_section_00 import assemble_matrix_2D_ex01 #---1 : In uniform mesh

assemble_matrix_2D   = compile_kernel( assemble_matrix_2D_ex01, arity=2)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2  = compile_kernel(assemble_norm_ex01, arity=1)

print('time to import utilities of Poisson equation =', time.time()-start)


from scipy.sparse import kron
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix, linalg as sla
from numpy import zeros, linalg, asarray
import numpy as np

from tabulate import tabulate

#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2 , V, u_p, xuh, periodic):

       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition
       u  = StencilVector(V.vector_space)
       # ...
       stiffness        = assemble_matrix_2D(V)
       stiffness        = apply_dirichlet(V, stiffness, dirichlet = [False, True])
       # ...
       M                = stiffness.tosparse()
       M                = apply_periodic(V, stiffness, periodic)
       lu               = sla.splu(csc_matrix(M))
       #---- assemble rhs
       rhs              = assemble_rhs( V, fields = [u_p])
       rhs              = apply_dirichlet(V, rhs, dirichlet = [False, True])
       # ...
       b                = apply_periodic(V, rhs, periodic)
       #--Solve a linear system
       xk               = lu.solve(b)
       x                = xk.reshape((V1.nbasis-V1.degree, V2.nbasis))
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
plt.savefig('periodic_Poisson')
plt.show()