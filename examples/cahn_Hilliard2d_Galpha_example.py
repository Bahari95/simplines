"""
cahn_Hilliard2d_Galpha_example.py

Generalized Alpha Method for Adaptive Time Step

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


import matplotlib                   as     mpl
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
import time

# --- for initialisation
from gallery_section_08 import assemble_massmatrix1D
from gallery_section_08 import assemble_vector_ex02 
# ---
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_dtrhs       = compile_kernel(assemble_vector_ex02, arity=1)

#... for G-alpha method
from gallery_section_08 import assemble_matrix_ex03 
from gallery_section_08 import assemble_vector_ex03
from gallery_section_08 import assemble_norm_ex01 

assemble2_stiffness = compile_kernel(assemble_matrix_ex03, arity=2)
assemble2_rhs       = compile_kernel(assemble_vector_ex03, arity=1)
assemble_norm_l2    = compile_kernel(assemble_norm_ex01, arity=1)


from scipy.sparse import kron
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix, linalg as sla

from numpy import zeros, linalg, asarray, linspace
import numpy as np

from tabulate import tabulate

#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#==============================================================================
#.......Poisson ALGORITHM
def Proj_solve(V1, V2 , V, alpha):
       periodic = [True, True]
       #... We delete the first and the last spline function
       #. as a technic for applying Dirichlet boundary condition
       u                = StencilVector(V.vector_space)
       dtu              = StencilVector(V.vector_space)

       M1               = assemble_mass1D(V1)
       M1               = apply_periodic(V1, M1)
       M1               = csr_matrix(M1)

       M2               = assemble_mass1D(V2)
       M2               = apply_periodic(V1, M2)
       M2               = csr_matrix(M2)
       # ... assemble 2D mass matrix from 1D mass matrix
       M                = kron(M1, M2)
 
       #---- assemble random control Points
       xh               = (np.random.rand(V1.nbasis-V1.degree,V2.nbasis-V2.degree)-1.)*0.05 +0.63
       xh               = apply_periodic(V, xh, periodic, update = True)
       u.from_array(V, xh)

       #---- assemble rhs of dt(u) at t = 0 from Cahn-Haliard equation
       rhs               = assemble_dtrhs( V, fields = [u], value =[alpha])
       rhs               = apply_periodic( V, rhs, periodic)
       # ...
       b                = -1.*rhs
       #--Solve a linear system
       lu               = sla.splu(csc_matrix(M))
       dtxh             = lu.solve(b)
       # .... reshape and apply periodic boundary condition
       dtxh             = dtxh.reshape((V1.nbasis-V1.degree, V2.nbasis-V2.degree))
       dtxh             = apply_periodic(V, dtxh, periodic, update = True)
       dtu.from_array(V, dtxh)
       # ... GL-FREE-ENERGY
       Norm             = assemble_norm_l2(V, fields=[u], value = [alpha])
       norm             = Norm.toarray()[0]
       return u, xh, dtu, dtxh, M, norm

#============================================================
#... two-stage pre-dictor–multicorrector algorithm
def Cahn_Hliard_solve(V1, V2, V, u, ut, xh, txh, dt, N_iter = None):

       periodic = [True, True]
       rho_inf    = 0.5
       alpha_m    = 0.5 * ((3. - rho_inf)/(1. + rho_inf))
       alpha_f    = 1/(1. + rho_inf)
       gamma      = 0.5 + alpha_m - alpha_f

       if N_iter is None:
          N_iter  = 100
       tol        = 1e-5
       
       u_m        = StencilVector(V.vector_space)
       u_f        = StencilVector(V.vector_space)
       
       #... Step 1 : Initialization
       xu_n       = zeros(V.nbasis)
       xu_n[:,:]  = xh[:,:]
       #...
       xtu_n      = zeros(V.nbasis)
       xtu_n[:,:] = ( (gamma-1.)/gamma ) * txh[:,:]
           
       #... step 2 : Multicoerrector :
       for i in range(0, N_iter): 
             
          #___ step (a)
          u_m.from_array(V, txh + alpha_m * (xtu_n - txh) )   
          u_f.from_array(V, xh  + alpha_f * (xu_n  -  xh) )
          #___ step (b): Genralized alpha level
           
          stiffness  = assemble2_stiffness(V, fields=[u_f], value = [dt, alpha])
          M          = apply_periodic(V, stiffness, periodic)
          #...
          rhs        = assemble2_rhs( V, fields=[u_f, u_m], value = [alpha])
          rhs        = apply_periodic(V, rhs, periodic)

          #--Solve a linear system
          b          = -1.*rhs
          #++ 
          lu         = sla.splu(csc_matrix(M))
          d_tx       = lu.solve(b)
          #print('CPU-time  SUP_LU== ', time.time()- start)
          d_tx       = d_tx.reshape((V1.nbasis-V1.degree, V2.nbasis-V2.degree))                    
          d_tx       = apply_periodic(V, d_tx, periodic, update= True)
          #___ step (c): update
          xtu_n[:,:] = xtu_n[:,:] + d_tx[:,:]
          xu_n[:,:]  = xu_n[:,:]  +  gamma * dt * d_tx[:,:]
          Res        = np.max(np.absolute(b))
          if Res < tol or Res > 1e10:
             break
       print('Iteration number : = {} Residual  = {}'.format( i, Res))
       u.from_array(V, xu_n)
       ut.from_array(V, xtu_n)
       # ... GL-FREE-ENERGY
       Norm             = assemble_norm_l2(V, fields=[u], value = [alpha])
       norm             = Norm.toarray()[0]
       return u, xu_n, ut, xtu_n, Res, norm

degree          = 2
nelements       = 64
# ...
dt              = 1e-8
t               = 0.
levels          = list(linspace(-0.15,1.15,100))
# ...
nbpts           = 80     # for plot
ii_max          = 100000 # for time iter
alpha           = 6000
stat_mement     = []
GL_free_energy  = []
n_iter          = []

#-------------------------------------------
# create the spline space for each direction
VP1        = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, periodic=True)
VP2        = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, periodic=True)
# create the tensor space
VPh        = TensorSpace(VP1, VP2)

#-------------------------------------------
# Initialisation
u_ch, xu_ch, dt_u_ch, dt_xu_ch, M_ms, norm = Proj_solve(VP1, VP2, VPh, alpha)
#  ...
#..we concerve the first random solution to comput statistical moment
u_Pr0      = xu_ch

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

a      = 0
i_save = 0
for ii in range(0, ii_max):
   t           += dt
   print('computation in time ================================', t)
   #-------------------------------------------
   u_ch, xu_ch, dt_u_ch, dt_xu_ch , Res, norm = Cahn_Hliard_solve(VP1, VP2, VPh, u_ch, dt_u_ch, xu_ch, dt_xu_ch, dt, alpha)
   
   #------------
   if  Res > 1e10 :
        print("Sorry. Your settings or the regularity assumption are not working !!!")
        break
   #if ii == a :
   if True :
	   a += 10        
	   #+++++++++++++++++++++++++++++
	   u_Pr = pyccel_sol_field_2d((nbpts,nbpts),  xu_ch, VPh.knots, VPh.degree)[0]
	   #Sol_CH.append( u_Pr)
	      
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
	   plt.savefig('figs/u_{}.png'.format(i_save))
	   plt.show(block=False)
	   plt.close()
	   i_save += 1

if True :    
  import imageio  
  with imageio.get_writer('cahn_haliard.gif', mode='I') as writer:  
      for filename in ['figs/u_{}.png'.format(i) for i in range(1,i_save,35)]:  
          image = imageio.imread(filename)  
          writer.append_data(image) 
