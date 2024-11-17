"""
poisson3d_example1.py

A basic test for periodic boundary conditions. (3D)

author :  M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet, apply_periodic

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_3d

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
import argparse

start = time.time()

from gallery_section_00 import assemble_stiffnessmatrix1D
from gallery_section_00 import assemble_massmatrix1D

assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

#---In Poisson equation
from gallery_section_00 import assemble_vector_ex02 #---1 : In uniform mesh
from gallery_section_00 import assemble_norm_ex02 #---1 : In uniform mesh

assemble_rhs         = compile_kernel(assemble_vector_ex02, arity=1)
assemble_norm_l2  = compile_kernel(assemble_norm_ex02, arity=1)
#<<<<<<<<<<<<<
#/: 
from gallery_section_00 import assemble_vector_ex20
from gallery_section_00 import assemble_vector_ex21
assemble_rhsy0      = compile_kernel(assemble_vector_ex20, arity=1)
assemble_rhsy1      = compile_kernel(assemble_vector_ex21, arity=1)
print('time to import utilities of Poisson equation =', time.time()-start)


from scipy.sparse import kron
from scipy.sparse import csr_matrix
from simplines import Poisson
from scipy.sparse import csc_matrix, linalg as sla
from numpy import zeros, linalg, asarray
import numpy as np

from tabulate import tabulate

#==============================================================================
#.......Poisson ALGORITHM
class poisson(object):

   def __init__(self, V1, V2, V3, V, periodic) :
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

       # Stiffness and Mass matrix in 1D in the third deriction
       K3 = assemble_stiffness1D(V3)
       K3 = apply_periodic(V3, K3)
       K3 = csr_matrix(K3)

       M3 = assemble_mass1D(V3)
       M3 = apply_periodic(V3, M3)
       M3 = csr_matrix(M3)
       
       mats_1 = [M1, K1]
       mats_2 = [M2, K2]
       mats_3 = [M3, K3]
       # ... for the projection
       mats_4 = [0.5*M1, 0.5*M1]
       mats_5 = [    M3,     M3]
       
       # ...
       self.poisson  = Poisson(mats_1, mats_2, mats_3)
       self.project  = Poisson(mats_4, mats_5)
       self.spaces   = [V1, V2, V3, V]
       self.periodic = periodic

   def Proj(self):
       [V1, V2, V3, V]  = self.spaces
       #... 
       V13      = TensorSpace(V1, V3)
       periodic = [self.periodic[0], self.periodic[2]]
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition
       u  = StencilVector(V.vector_space)
       x                = zeros(V.nbasis)
       # ... y = min_value
       #---- assemble rhs
       rhs              = assemble_rhsy0(V13)
       b                = apply_periodic(V13, rhs, periodic)
       #--Solve a linear system
       xk               = self.project.solve(b)
       xk               = xk.reshape((V1.nbasis-V1.degree, V3.nbasis-V3.degree))
       xk0              = apply_periodic(V13, xk, periodic, update = True)       

       # ... y = max_value
       #---- assemble rhs
       rhs              = assemble_rhsy1(V13)
       b                = apply_periodic(V13, rhs, periodic)
       #--Solve a linear system
       xk               = self.project.solve(b)
       xk               = xk.reshape((V1.nbasis-V1.degree, V3.nbasis-V3.degree))
       xk1              = apply_periodic(V13, xk, periodic, update = True)       
       #..
       x[:, 0, :]       = xk0[:,:]  
       x[:,-1, :]       = xk1[:,:]
       u.from_array(V, x)
       return x, u
          
   def solve(self, xuh, u_p):
   
       [V1, V2, V3, V]  = self.spaces
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition
       u  = StencilVector(V.vector_space)
      
       #---- assemble rhs
       rhs              = assemble_rhs( V, fields = [u_p])
       # ...
       b                = apply_periodic(V, rhs, periodic)
       b                = b.reshape(V1.nbasis-V1.degree,V2.nbasis, V3.nbasis-V3.degree)
       b                = b[:, 1:-1, :]       
       b                = b.reshape((V1.nbasis-V1.degree)*(V2.nbasis-2)*(V3.nbasis-V3.degree))
       #--Solve a linear system
       xk               = self.poisson.solve(b)
       xk               = xk.reshape((V1.nbasis-V1.degree, V2.nbasis-2, V3.nbasis-V3.degree))
       x                = zeros((V1.nbasis-V1.degree,V2.nbasis, V3.nbasis-V3.degree))
       x[:,1:-1,:]      = xk[:,:,:]
       x                = apply_periodic(V, x, self.periodic, update = True)       
       x[:,0,:]         = xuh[:,0,:]
       x[:,-1,:]        = xuh[:,-1,:]
       u.from_array(V, x)

       Norm    = assemble_norm_l2(V, fields=[u])
       norm    = Norm.toarray()
       l2_norm = norm[0]
       H1_norm = norm[1]
       
       return u, x, l2_norm, H1_norm

degree     = 3
nelements  = 64
periodic   = [True, False, True]
#---------------------------------------
from numpy import sin, exp, pi, linspace
u_exact = lambda x, y, z : x**2*sin(5.*pi*y)*z**2

#-------------------------------------------
# create the spline space for each direction
grids     = linspace(-0.5*pi, 0.5*pi,  nelements + 1)
V1 = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2, periodic=periodic[0])
#...
grids     = linspace(-1., 1.,  nelements + 1)
V2        = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2, periodic = periodic[1])
#...
grids     = linspace(-0.5*pi, 0.5*pi,  nelements + 1)
V3 = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2, periodic=periodic[2])

# create the tensor space
Vh = TensorSpace(V1, V2, V3)
#.. Initialisation of tools to solve the problem
Ps = poisson(V1, V2, V3, Vh, periodic)

# ... Dirichlet boundary condition 
x0uh, u_p =  Ps.Proj()

print('#---IN-UNIFORM--MESH')
u_ph, xuh, l2_norm, H1_norm = Ps.solve(x0uh, u_p)
print('l2_norm = {} H1_norm = {} '.format(l2_norm, H1_norm) ) 
nbpts    = 80

print(xuh.shape)
u_poisson, u_xpoisson, u_ypoisson, u_zpoisson, X, Y, Z = pyccel_sol_field_3d((nbpts,nbpts,nbpts),  xuh , Vh.knots, Vh.degree)


def main():
    parser = argparse.ArgumentParser(description="A script with a plot option.")
    parser.add_argument("--plot", action="store_true", help="Run the plotting code")
    args = parser.parse_args()

    if args.plot:
        print("Plotting code is running...")
        # Insert your plotting code here
        np.save('figs/mesh_x.npy',X)
        np.save('figs/mesh_y.npy',Y)
        np.save('figs/mesh_z.npy',Z)
        np.save('figs/Ex_sol.npy',   u_exact(X,Y,Z).T)
        np.save('figs/appr_sol.npy',   u_poisson)
        np.save('figs/appr_soldx.npy',u_xpoisson)
        np.save('figs/appr_soldy.npy',u_ypoisson)
        np.save('figs/appr_soldz.npy',u_zpoisson)

        # ... code for paraview
        from evtk.hl import gridToVTK 
        import numpy as np 
        import random as rnd 

        # Coordinates
        x = np.load("figs/mesh_x.npy")
        y = np.load("figs/mesh_y.npy")
        z = np.load("figs/mesh_z.npy")
        # We add Jacobian function to make the grid more interesting
        Ex_sol       = np.load("figs/Ex_sol.npy")
        appr_sol     = np.load("figs/appr_sol.npy")
        appr_soldx   = np.load("figs/appr_soldx.npy")
        appr_soldy   = np.load("figs/appr_soldy.npy")
        appr_soldz   = np.load("figs/appr_soldz.npy")
        # Dimensions 
        nx, ny, nz = x.shape[0], y.shape[0], z.shape[0]
        ncells = nx * ny * nz
        npoints = (nx + 1) * (ny + 1) * (nz + 1)

        # Variables <br>
        gridToVTK("./domain", x, y, z, pointData = {"Ex_sol" : Ex_sol, "appr_sol" : appr_sol,"dx_appr_sol" : appr_soldx,"dy_appr_sol" : appr_soldy,"dz_appr_sol" : appr_soldz,})
    else:
        print("No plotting requested.")

if __name__ == "__main__":
    main()