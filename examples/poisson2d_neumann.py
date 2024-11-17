"""
poisson2d_neumann.py

A basic test for Neumann boundary conditions.(2D)

author :  M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

# .. Matrices in 1D ..
from gallery_section_02 import assemble_stiffnessmatrix1D
from gallery_section_02 import assemble_massmatrix1D
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

#---In Poisson equation
from gallery_section_02 import assemble_vector_ex01    #---1 : In uniform mesh
from gallery_section_02 import assemble_norm_ex01      #---1 : In uniform mesh

assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
#..
from   scipy.sparse                 import kron
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   tabulate                     import tabulate
import numpy                        as     np
import timeit
import time

from tabulate import tabulate


#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2, V):
       u                   = StencilVector(V.vector_space)
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       K1                  = assemble_stiffness1D(V1)
       K1                  = K1.tosparse()
       K1                  = K1.toarray()[1:-1,1:-1]
       K1                  = csr_matrix(K1)

       M1                  = assemble_mass1D(V1)
       M1                  = M1.tosparse()
       M1                  = M1.toarray()[1:-1,1:-1]
       M1                  = csr_matrix(M1)

       # Stiffness and Mass matrix in 1D in the second deriction
       K2                  = assemble_stiffness1D(V2)
       K2                  = K2.tosparse()
       K2                  = K2.toarray()
       K2                  = csr_matrix(K2)

       M2                  = assemble_mass1D(V2)
       M2                  = M2.tosparse()
       M2                  = M2.toarray()
       M2                  = csr_matrix(M2)
       
       # Stiffness and Mass matrix in 1D in the thrd deriction
       M                   = kron(K1,M2)+kron(M1,K2)+kron(M1,M2)
       lu                  = sla.splu(csc_matrix(M))
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V, fields=[u] )
       b                   = rhs.toarray()
       b                   = b.reshape(V.nbasis)
       b                   = b[1:-1, :]      
       b                   = b.reshape((V1.nbasis-2)*(V2.nbasis))
       # ...
       xkron               = lu.solve(b)       
       xkron               = xkron.reshape([V1.nbasis-2,V2.nbasis])
       # ...
       x                   = np.zeros(V.nbasis)
       x[1:-1,  :]         = xkron
       x                   = x.reshape(V.nbasis)
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

degree      = 2
quad_degree = degree + 1

#----------------------
#..... Initialisation and computing optimal mapping for 16*16
#----------------------
nelements  = 64
# create the spline space for each direction
V1   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V2   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
Vh   = TensorSpace(V1, V2)

print('#---IN-UNIFORM--MESH')
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, Vh)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm, H1_norm))

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
Sol = lambda x, y : sin(pi*x)* sin(pi*y)
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
plt.savefig('Poisson3D.png')
plt.show()
