"""
poisson1d_example.py

A basic test for poisson equation.(1D)

author :  M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_3d

# .. Matrices in 1D ..
from gallery_section_01 import assemble_stiffnessmatrix1D
from gallery_section_01 import assemble_massmatrix1D
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

#---In Poisson equation
from gallery_section_01 import assemble_vector_ex01    #---1 : In uniform mesh
from gallery_section_01 import assemble_norm_ex01      #---1 : In uniform mesh

assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_norm_l2     = compile_kernel(assemble_norm_ex01, arity=1)

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
import numpy                        as     np
font = {'family': 'serif', 
	 'color':  'k', 
	 'weight': 'normal', 
	 'size': 25, 
		 } 
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
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V):
       u                   = StencilVector(V.vector_space)
       # ++++
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       K1                  = assemble_stiffness1D(V)
       K1                  = K1.tosparse()
       K1                  = K1.toarray()[1:-1,1:-1]
       #K1                  = csc_matrix(K1)

       M1                  = assemble_mass1D(V)
       M1                  = M1.tosparse()
       M1                  = M1.toarray()[1:-1,1:-1]
       M1                  = csr_matrix(M1)

       # ...
       #M                   = kron(K1,kron(M2,M3))+kron(M1,kron(K2,M3))+kron(M1,kron(M2,K3))
       #lu                  = sla.splu(csc_matrix(K1))
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V )
       b                   = rhs.toarray()
       b                   = b.reshape(V.nbasis)
       b                   = b[1:-1]      
       # ...
       #xkron               = lu.solve(b)       
       xkron, status = sla.cg(K1, b, tol=1.e-17, maxiter=5000)
       # ...
       x                   = np.zeros(V.nbasis)
       x[1:-1]             = xkron
       u.from_array(V, x)
       # ...
       Norm                = assemble_norm_l2(V, fields=[u]) 
       norm                = Norm.toarray()
       l2_norm             = norm[0]
       H1_norm             = norm[1]       
       return u, x, l2_norm, H1_norm

#degree      = 5
nb_max    = 10
ls_times  = [2**nb for nb in range(4,nb_max)]
fig       = plt.figure() 
for degree in range(3,9):
	error_L2 = []
	error_H1 = []
	#----------------------
	#..... Initialisation and computing optimal mapping for 16*16
	#----------------------
	for nb in range( 4,nb_max):
		nelements = 2**nb
		# create the spline space for each direction
		V1   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2)

		print('#---IN-UNIFORM--MESH')
		u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1)
		print('degree = {}-----> L^2-error ={} -----> H^1-error = {}'.format(degree, l2_norm, H1_norm))
		error_L2.append(l2_norm)
		error_H1.append(H1_norm)
	plt.plot(ls_times, error_L2, '*-k', linewidth = 2., label='$\mathbf{}$'.format(degree))
print('message')
plt.yscale('log')
#plt.xscale('log')
#plt.grid(color='k', linestyle='--', linewidth=0.5, which ="both")
#plt.xlabel('$\mathbf{time}$',  fontweight ='bold', fontdict=font)
#plt.ylabel('$\mathbf{L^2}$-norm',  fontweight ='bold', fontdict=font)
#plt.legend(fontsize="15")
#fig.tight_layout()
plt.savefig('figs/Error')
plt.show(block=True)
plt.close()
