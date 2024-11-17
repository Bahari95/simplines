"""
poisson3d_dirichlet.py

author :  M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_3d

# .. Matrices in 1D ..
from gallery_section_03 import assemble_stiffnessmatrix1D
from gallery_section_03 import assemble_massmatrix1D
from gallery_section_03 import assemble_matrix_ex11
from gallery_section_03 import assemble_matrix_ex12
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex11, arity=2)
assemble_matrix_ex10 = compile_kernel(assemble_matrix_ex12, arity=2)

#---In Poisson equation
from gallery_section_03 import assemble_vector_ex01    #---1 : In uniform mesh
from gallery_section_03 import assemble_matrix_un_ex01 #---1 : In uniform mesh
from gallery_section_03 import assemble_norm_ex01      #---1 : In uniform mesh

assemble_stiffness2D = compile_kernel(assemble_matrix_un_ex01, arity=2)
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
def poisson_solve(V1, V2, V3, V):
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
       K2                  = K2.toarray()[1:-1,1:-1]
       K2                  = csr_matrix(K2)

       M2                  = assemble_mass1D(V2)
       M2                  = M2.tosparse()
       M2                  = M2.toarray()[1:-1,1:-1]
       M2                  = csr_matrix(M2)
       
       # Stiffness and Mass matrix in 1D in the thrd deriction
       K3                  = assemble_stiffness1D(V3)
       K3                  = K3.tosparse()
       K3                  = K3.toarray()[1:-1,1:-1]
       K3                  = csr_matrix(K3)

       M3                  = assemble_mass1D(V3)
       M3                  = M3.tosparse()
       M3                  = M3.toarray()[1:-1,1:-1]
       M3                  = csr_matrix(M3)

       # ...
       M                   = kron(K1,kron(M2,M3))+kron(M1,kron(K2,M3))+kron(M1,kron(M2,K3))
       lu                  = sla.splu(csc_matrix(M))
       # ++++
       #--Assembles a right hand side of Poisson equation
       rhs                 = assemble_rhs( V )
       b                   = rhs.toarray()
       b                   = b.reshape(V.nbasis)
       b                   = b[1:-1, 1:-1, 1:-1]      
       b                   = b.reshape((V1.nbasis-2)*(V2.nbasis-2)*(V3.nbasis-2))
       # ...
       xkron               = lu.solve(b)       
       xkron               = xkron.reshape([V1.nbasis-2,V2.nbasis-2,V3.nbasis-2])
       # ...
       x                   = np.zeros(V.nbasis)
       x[1:-1, 1:-1, 1:-1] = xkron
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
nelements  = 16
# create the spline space for each direction
V1   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V2   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V3   = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
V    = TensorSpace(V1, V2, V3)

print('#---IN-UNIFORM--MESH')
u_pH, xuh, l2_norm, H1_norm = poisson_solve(V1, V2, V3, V)
print('-----> L^2-error ={} -----> H^1-error = {}'.format(l2_norm, H1_norm))

#---Compute a solution
nbpts = 100
# # ........................................................
# ....................For testing in one nelements
# #.........................................................
if True :
	#---Compute a solution
	u,   ux, uy, uz, X, Y, Z      = pyccel_sol_field_3d((nbpts, nbpts, nbpts),  xuh,   V.knots, V.degree)
	
	solut = lambda t, x, y : sin(pi*t)*x**2*y*3*sin(4.*pi*(1.-x))*(1.-y) 
	#solut = lambda t, x, y :  sin( pi*t)* sin( pi*x)* sin( pi*y)
	
	# set up a figure twice as wide as it is tall
	fig = plt.figure(figsize=plt.figaspect(0.5))
	#===============
	# First subplot
	# set up the axes for the first plot
	ax = fig.add_subplot(1, 2, 1, projection='3d')
	# plot a 3D surface like in the example mplot3d/surface3d_demo
	surf0 = ax.plot_surface(Y[0,:,:], Z[0,:,:], u[50,:,:], rstride=1, cstride=1, cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate solution in uniform mesh')
	ax.set_xlabel('X',  fontweight ='bold')
	ax.set_ylabel('Y',  fontweight ='bold')
	# Add a color bar which maps values to colors.
	fig.colorbar(surf0, shrink=0.5, aspect=25)

	#===============
	# Second subplot
	ax = fig.add_subplot(1, 2, 2, projection='3d')
	surf = ax.plot_surface(Y[0,:,:], Z[0,:,:], solut(0.5,Y[0,:,:], Z[0,:,:]), cmap=cm.coolwarm,
		               linewidth=0, antialiased=False)
	ax.set_xlim(0.0, 1.0)
	ax.set_ylim(0.0, 1.0)
	ax.zaxis.set_major_locator(LinearLocator(10))
	ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
	#ax.set_title('Approximate Solution in adaptive meshes')
	ax.set_xlabel('F1',  fontweight ='bold')
	ax.set_ylabel('F2',  fontweight ='bold')
	fig.colorbar(surf, shrink=0.5, aspect=25)
	plt.savefig('Poisson3D.png')
	plt.show()
