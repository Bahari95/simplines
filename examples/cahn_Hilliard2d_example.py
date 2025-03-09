"""
cahn_Hilliard2d_example.py

implicit_scheme

author :  M. BAHARI
"""
from simplines import compile_kernel, apply_dirichlet, apply_periodic

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d

import matplotlib                   as     mpl
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
import time

# .. for the initialisation
from examples.gallery.gallery_section_09 import assemble_massmatrix1D
from examples.gallery.gallery_section_09 import assemble_vector_ex01 #---1 : Projection L2
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)

# ---. for cahn-haliard
from examples.gallery.gallery_section_09 import assemble_matrix_ex03 
from examples.gallery.gallery_section_09 import assemble_vector_ex03
from examples.gallery.gallery_section_09 import assemble_norm_ex01 

assemble2_stiffness = compile_kernel(assemble_matrix_ex03, arity=2)
assemble2_rhs       = compile_kernel(assemble_vector_ex03, arity=1)
assemble_norm_l2    = compile_kernel(assemble_norm_ex01, arity=1)

# ---
from scipy.sparse        import kron
from scipy.sparse        import csr_matrix
from scipy.sparse        import csc_matrix, linalg as sla
from numpy               import zeros, linalg, asarray, linspace
#++
import numpy as np
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists


#==============================================================================
#.......Poisson ALGORITHM
def Proj_solve(V1, V2 , V, alpha):
       # ... periodic boundary in all directions
       periodic = [True, True]

       u                 = StencilVector(V.vector_space)
       #dtu              = StencilVector(V.vector_space)

       M1                = assemble_mass1D(V1)
       M1                = apply_periodic(V1, M1)
       M1                = csr_matrix(M1)

       M2                = assemble_mass1D(V2)
       M2                = apply_periodic(V1, M2)
       M2                = csr_matrix(M2)
       
       M_res             = kron(M1,M2)
       # ... assemble random control Points
       xh                = (np.random.rand(V1.nbasis-V1.degree,V2.nbasis-V2.degree)-1.)*0.05 +0.63
       xh                = apply_periodic(V, xh, periodic, update = True)
       u.from_array(V, xh)
       # ... GL-FREE-ENERGY
       Norm             = assemble_norm_l2(V, fields=[u], value = [alpha])
       norm             = Norm.toarray()[0]
       return u, xh,  M_res, norm

#============================================================
#... two-stage pre-dictor–multicorrector algorithm
def Cahn_Hliard_solve(V1, V2, V, u, xh, dt, alpha, N_iter = None):
       
       # ... periodic boundary in all directions
       periodic = [True, True]

       if N_iter is None:
          N_iter  = 100
       tol        = 1e-7
       
       u_f        = StencilVector(V.vector_space)
       
       #... Step 1 : Initialization
       xu_n       = zeros(V.nbasis)
       xu_n[:,:]  = xh[:,:]
       
       # ...    
       u_f.from_array(V, xu_n )
       #... step 2 : Multicoerrector :
       for i in range(0, N_iter): 
             
          #___ step (b): Genralized alpha level
           
          stiffness  = assemble2_stiffness(V, fields=[u_f], value = [dt, alpha])
          M          = apply_periodic(V, stiffness, periodic)
          #...
          rhs        = assemble2_rhs( V, fields=[u_f, u], value = [dt, alpha])
          rhs        = apply_periodic(V, rhs, periodic)
          # ---
          b          = -1.*rhs
          #--Solve a linear system
          lu         = sla.splu(csc_matrix(M))
          d_thx      = lu.solve(b)
          # --- update and apply periodic boundary conditions
          d_tx       = d_thx.reshape((V1.nbasis-V1.degree, V2.nbasis-V2.degree))                    
          d_tx       = apply_periodic(V, d_tx, periodic, update= True)
          #___ step (c): update
          xu_n[:,:] = xu_n[:,:] + d_tx[:,:]
          Res        = np.max(np.absolute(d_thx))
          #___ step (d)
          u_f.from_array(V, xu_n )
          if Res < tol or Res > 1e3:
             break
       u.from_array(V, xu_n)
       print('perform the iteration number : = {} Residual  = {}'.format( i, Res))
       # ... GL-FREE-ENERGY
       Norm             = assemble_norm_l2(V, fields=[u], value = [alpha])
       norm             = Norm.toarray()[0]
       return u, xu_n, Res, norm

degree          = 2
nelements       = 32
# ...
dt              = 1e-8
alpha           = 6000
t               = 0.
levels          = list(linspace(-0.1,1.1,100))
# ...
nbpts           = 100    # for plot
ii_max          = 10000  # for time iter
stat_mement     = []
GL_free_energy  = []
n_iter          = []

#-------------------------------------------
# create the spline space for each direction
grids      = linspace(0., 1.,  nelements + 1)
VP1        = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2, periodic=True)
VP2        = SplineSpace(degree=degree, nelements= nelements, grid = grids, nderiv = 2, periodic=True)
# create the tensor space
VPh        = TensorSpace(VP1, VP2)

#-------------------------------------------
# --- Initialisation
u_ch, xu_ch, M_ms, norm = Proj_solve(VP1, VP2, VPh, alpha)
# ...
u_Pr0 = xu_ch

#-------------------------------------------
u_Pr, u_xPr, u_yPr, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  xu_ch, VPh.knots, VPh.degree)

if True :
   #-------------------------------------------------------------------------------------------
   u_Pr, u_xPr, u_yPr, Y, X = pyccel_sol_field_2d((nbpts,nbpts),  xu_ch, VPh.knots, VPh.degree)
   u_Pr = u_Pr.T
   #+++++++++++++++++++++++++++++
   du_ch     = (u_Pr0[:-degree,:-degree]-xu_ch[:-degree,:-degree]).reshape((VP1.nbasis-degree)*(VP2.nbasis-degree))
   stat_mement.append((M_ms.dot(du_ch)).dot(du_ch) )
   GL_free_energy.append(norm)
   n_iter.append(t)   
   #+++-----------------------------------------------------------------------------------
   # ... Statistical moment
   plt.figure() 
   plt.subplot(121)
   plt.title( '$\mathbf{||c-c_0||_{L^2}}$')
   plt.plot(n_iter, stat_mement, 'o-b', linewidth = 2.)
   plt.xscale('log')
   plt.xlabel('time',  fontweight ='bold')
   plt.grid(True)
   #plt.legend()
   
   axes = plt.subplot(122)
   axes.set_aspect(1)
   plt.title( '$GL-Free-Energy$')
   plt.plot(n_iter, GL_free_energy,  'o--r', linewidth = 2.)
   plt.xscale('log')
   plt.xlabel('time',  fontweight ='bold')
   plt.grid(True)
   #plt.legend()
   plt.subplots_adjust(wspace=0.3)
   #plt.savefig('figs/Pu.png')
   plt.show(block=True)
   #plt.pause(0.3)
   plt.close()
   # ...
   figtitle        = 'Cahn_haliard_equation'
   fig, axes       = plt.subplots( 1, 1, figsize=[12,12], num=figtitle )
   axes.set_aspect('equal')
   axes.set_title( 'Approximate solution at t= {}'.format(t) )
   im2 = axes.contourf( X, Y, u_Pr, levels, cmap= 'jet')
   divider = make_axes_locatable(axes) 
   cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
   plt.colorbar(im2, cax=cax)
   
   fig.tight_layout()
   plt.subplots_adjust(wspace=0.3)
   #plt.savefig('figs/u_{}.png'.format(0))
   plt.show(block=True)
   #plt.pause(0.3)
   plt.close()


for ii in range(0, ii_max):
   # ... update time
   t           += dt
   print('In time : ', t)
   #-------------------------------------------
   u_ch, xu_ch, Res, norm = Cahn_Hliard_solve(VP1, VP2, VPh, u_ch, xu_ch, dt, alpha)
   
   #------------
   if  Res > 1e10 :
        print("Sorry. Your settings or the regularity assumption are not working !!!")
        break   

   #+++++++++++++++++++++++++++++
   u_Pr = pyccel_sol_field_2d((nbpts,nbpts),  xu_ch, VPh.knots, VPh.degree)[0]
   # ...
   du_ch     = (u_Pr0[:-degree,:-degree]-xu_ch[:-degree,:-degree]).reshape((VP1.nbasis-degree)*(VP2.nbasis-degree))
   stat_mement.append((M_ms.dot(du_ch)).dot(du_ch) )
   GL_free_energy.append(norm)
   n_iter.append(t)   
   #+++-----------------------------------------------------------------------------------
   # ... Statistical moment
   plt.figure() 
   plt.subplot(121)
   plt.title( '$\mathbf{||c-c_0||_{L^2}}$')
   plt.plot(n_iter, stat_mement, 'o-b', linewidth = 2.)
   plt.xscale('log')
   plt.xlabel('time',  fontweight ='bold')
   plt.grid(True)
   #plt.legend()
   
   axes = plt.subplot(122)
   axes.set_aspect(1)
   plt.title( '$GL-Free-Energy$')
   plt.plot(n_iter, GL_free_energy,  'o--r', linewidth = 2.)
   plt.xscale('log')
   plt.xlabel('time',  fontweight ='bold')
   plt.grid(True)
   #plt.legend()
   plt.subplots_adjust(wspace=0.3)
   plt.savefig('figs/Pu.png')
   plt.show(block=False)
   plt.close()
   # ...
   figtitle        = 'Cahn_haliard_equation'
   fig, axes       = plt.subplots( 1, 1, figsize=[12,12], num=figtitle )
   axes.set_aspect('equal')
   axes.set_title( 'Approximate solution at t= {}'.format(t) )
   im2 = axes.contourf( X, Y, u_Pr, levels, cmap= 'jet')
   divider = make_axes_locatable(axes) 
   cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
   plt.colorbar(im2, cax=cax)
   
   fig.tight_layout()
   plt.subplots_adjust(wspace=0.3)
   plt.savefig('figs/u_{}.png'.format(ii))
   plt.show(block=False)
   plt.close()

#... Turn pictures to .gif
if True :   
 import imageio 
 with imageio.get_writer('cahn_haliard.gif', mode='I') as writer: 
     for filename in ['figs/u_{}.png'.format(i) for i in range(1,ii_max)]: 
         image = imageio.imread(filename) 
         writer.append_data(image) 
