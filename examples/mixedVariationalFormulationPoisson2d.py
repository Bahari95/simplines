"""
mixedVariationalFormulationPoisson2d.py : unit-square

author :  M. BAHARI
"""
from simplines import compile_kernel,  apply_dirichlet

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import Poisson
from simplines import least_square_Bspline
#.. Prologation by knots insertion matrix
from simplines import prolongation_matrix

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
from examples.gallery.gallery_section_07 import assemble_stiffnessmatrix1D
from examples.gallery.gallery_section_07 import assemble_massmatrix1D
from examples.gallery.gallery_section_07 import assemble_matrix_ex01
from examples.gallery.gallery_section_07 import assemble_matrix_ex02
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex01, arity=1)
assemble_matrix_ex02 = compile_kernel(assemble_matrix_ex02, arity=1)

#---In Poisson equation
from examples.gallery.gallery_section_07 import assemble_vector_ex01 #---1 : In uniform mesh
assemble_rhs     = compile_kernel(assemble_vector_ex01, arity=1)
from examples.gallery.gallery_section_07 import assemble_norm_ex01 #---1 : In uniform mesh
assemble_norm_l2 = compile_kernel(assemble_norm_ex01, arity=1)
#print('time to import utilities of Poisson equation =', time.time()-start)

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
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#==============================================================================
#.......Poisson ALGORITHM
def poisson_solve(V1, V2, V3, V4, V,  V00, V11, V01, V10, u_01 = None, u_10 = None, x0 = None, y0 = None):

       u11   = StencilVector(V01.vector_space)
       u12   = StencilVector(V10.vector_space)
       #...step 3    
       x11 = np.zeros(V01.nbasis) # dx/ appr.solution
       x12 = np.zeros(V10.nbasis) # dy/ appr.solution
       #___
       I1  = np.eye(V3.nbasis)
       I2  = np.eye(V4.nbasis)

       #... We delete the first and the last spline function
       #. as a technic for applying Neumann boundary condition

       #..Stiffness and Mass matrix in 1D in the first deriction
       D1 = assemble_mass1D(V3)
       D1 = D1.tosparse()
       D1 = D1.toarray()
       D1 = csr_matrix(D1)
       #___
       M1 = assemble_mass1D(V1)
       M1 = M1.tosparse()
       m1 = M1
       M1 = M1.toarray()[1:-1,1:-1]
       M1 = csc_matrix(M1)
       m1 = csr_matrix(m1)

       # Stiffness and Mass matrix in 1D in the second deriction
       D2 = assemble_mass1D(V4)
       D2 = D2.tosparse()
       D2 = D2.toarray()
       D2 = csr_matrix(D2)
       #___
       M2 = assemble_mass1D(V2)
       M2 = M2.tosparse()
       m2 = M2
       M2 = M2.toarray()[1:-1,1:-1]
       M2 = csc_matrix(M2)
       m2 = csr_matrix(m2)

       #...
       R1 = assemble_matrix_ex01(V01)
       R1 = R1.toarray()
       R1 = R1.reshape(V01.nbasis)
       r1 = R1.T
       R1 = R1[1:-1,:].T
       R1 = csr_matrix(R1)
       r1 = csr_matrix(r1)
       #___
       R2 = assemble_matrix_ex02(V10)
       R2 = R2.toarray()
       R2 = R2.reshape(V10.nbasis)
       r2 = R2
       R2 = R2[:,1:-1]
       R2 = csr_matrix(R2)
       r2 = csr_matrix(r2)

       #...step 1
       M1 = sla.inv(M1)
       A1 = M1.dot(R1.T)
       K1 = R1.dot( A1)
       K1 = csr_matrix(K1)
       #___
       M2 = sla.inv(M2)
       A2 = M2.dot( R2.T)
       K2 = R2.dot( A2)
       K2 = csr_matrix(K2)

       #...step 2
       mats_1 = [D1, K1]
       mats_2 = [D2, K2]

       # ...Fast Solver
       poisson = Poisson(mats_1, mats_2)

       #---Assembles a right hand side of Poisson equation
       rhs = assemble_rhs( V11)
       b   = rhs.toarray()
       #...
       b01 = -kron(r1, D2).dot(x0.reshape(V1.nbasis*V3.nbasis))
       #__
       b10 = -kron(D1, r2).dot(y0.reshape(V4.nbasis*V2.nbasis) )
       b   = b01 + b10 + b  
       #...non homogenoeus Neumann boundary 
       b11   = -kron(m1[1:-1,:], D2).dot(x0.reshape(V1.nbasis*V3.nbasis))
       #___
       b12   = -kron(D1, m2[1:-1,:]).dot(y0.reshape(V4.nbasis*V2.nbasis))
       #___Solve first system
       r   =  kron(A1.T, I2).dot(b11) + kron(I1, A2.T).dot(b12) - b
       x2  = poisson.solve(r)

       #...
       D1 = sla.inv(csc_matrix(D1))
       D2 = sla.inv(csc_matrix(D2))
       #___
       x11_1         = kron(A1, I2)
       x11[1:-1,:]   = (kron(M1, D2).dot(b11) - x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])
       #___
       x12_1         =  kron(I1, A2)
       x12[:,1:-1]   =  (kron(D1, M2).dot(b12) - x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])

       #...Assembles Neumann boundary conditions
       x11[0, : ]                        = x0[0, : ]    
       x11[V1.nelements+V1.degree-1,:]   = x0[V1.nelements+V1.degree-1, : ]
       x12[:,0]                          = y0[:,0]
       x12[:,V2.nelements+V2.degree-1]   = y0[:,V2.nelements+V2.degree-1]
       u11.from_array(V01, x11)
       u12.from_array(V10, x12)

       #--Computes error l2 and H1
       Norm    = assemble_norm_l2(V01, fields=[u11, u12])
       norm    = Norm.toarray()
       H1_norm =  norm[1]
     
       return u11, u12, x2, x11, x12, H1_norm

degree    = 3
nelements = 64
#----------------------
# create the spline space for each direction
V1 = SplineSpace(degree=degree, nelements= nelements, nderiv = 1)
V2 = SplineSpace(degree=degree, nelements= nelements, nderiv = 1)
V3 = SplineSpace(degree=degree-1, nelements= nelements, nderiv = 1, mixed = True)
V4 = SplineSpace(degree=degree-1, nelements= nelements, nderiv = 1, mixed = True)

# create the tensor space
Vh00 = TensorSpace(V1, V2)
Vh11 = TensorSpace(V3, V4)
Vh01 = TensorSpace(V1, V3)
Vh10 = TensorSpace(V4, V2)
Vh   = TensorSpace(V1, V2, V3, V4)

#------------------------------
from numpy import cos, sin, pi, exp
fx0 = lambda y : 2.*0.*exp(1.-0.**2)*cos(pi*y**2)
fx1 = lambda y : -2.*1.*exp(1.-1.**2)*cos(pi*y**2)
#__
fy0 = lambda x :  2.*pi*x*exp(1.-x**2)*sin(pi*0.**2)
fy1 = lambda x :  -2.*pi*x*exp(1.-x**2)*sin(pi*1.**2)

u01   = StencilVector(Vh01.vector_space)
u10   = StencilVector(Vh10.vector_space)

xD = np.zeros(Vh01.nbasis)
yD = np.zeros(Vh10.nbasis)

X0 = least_square_Bspline(V3.degree, V3.knots, fx0)
X1 = least_square_Bspline(V3.degree, V3.knots, fx1)
Y0 = least_square_Bspline(V4.degree, V4.knots, fy0)
Y1 = least_square_Bspline(V4.degree, V4.knots, fy1)

xD[0, : ]                   = X0
xD[nelements+degree-1, : ]  = X1
yD[:,0]                     = Y0
yD[:, nelements+degree - 1] = Y1
u01.from_array(Vh01, xD)
u10.from_array(Vh10, yD)

#+++++++++++++++++++++++++++++++++++
print('++Mixed-formulation-for-Poisson--In-Uniform--Mesh')
start = time.time()
u11_pH, u12_pH, x2uh, x11uh, x12uh, H1_norm = poisson_solve(V1, V2, V3, V4, Vh, Vh00, Vh11, Vh01, Vh10, u_01 = u01, u_10 = u10, x0 = xD, y0 = yD)
cpu_time =  time.time()-start

print('->  degree = {}  nelement = {}   error_H1 = {}  CPU-time = {}'.format(degree, nelements, H1_norm, cpu_time))
#---Compute a solution
nbpts = 60
#---Solution in uniform mesh
#u  = pyccel_sol_field_2d((nbpts,nbpts),  x2uh , V11.knots, V11.degree)[1]
u, ux, uy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x2uh.reshape(Vh11.nbasis), Vh11.knots, Vh11.degree)
ux = pyccel_sol_field_2d((nbpts,nbpts),  x11uh, Vh01.knots, Vh01.degree)[0]
uy = pyccel_sol_field_2d((nbpts,nbpts),  x12uh, Vh10.knots, Vh10.degree)[0]
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
from numpy import pi, sin, cos
sol = lambda x,y : exp(1.-x**2)*cos(pi*y**2)
sol_dx = lambda x,y : -2.*x*exp(1.-x**2)*cos(pi*y**2)
sol_dy = lambda x,y : -2.*pi*y*exp(1.-x**2)*sin(pi*y**2)

figtitle        = 'mixed_error_dx'
fig, axes       = plt.subplots( 1, 2, figsize=[12,12], gridspec_kw={'width_ratios': [2, 2]} , num=figtitle )
for ax in axes:
   ax.set_aspect('equal')

#axes[0].set_title( 'in uniform mesh ' )
im = axes[0].contourf( X, Y, ux, np.linspace(np.min(ux),np.max(ux),100), cmap= 'jet')
divider = make_axes_locatable(axes[0]) 
cax     = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im, cax=cax) 
Z = sol_dx(X,Y)
ima = axes[1].contourf( X, Y, Z, np.linspace(np.min(Z),np.max(Z),100), cmap= 'jet')
divider = make_axes_locatable(axes[1]) 
cax     = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(ima, cax=cax) 
fig.tight_layout()
plt.savefig('figs/density_function.png')
plt.show(block=False)
plt.close()
