"""
elasticity.py

# elasticity example :Infinite plate with circular hole under constant in-plane tension in the x-direction

@author : M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet, apply_zeros

#from spaces import SplineSpace
#from spaces import TensorSpace

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import pyccel_sol_field_2d
from simplines import prolongation_matrix
from simplines import getGeometryMap

import time

#--- In Elasticity equation
from gallery_section_05 import assemble_matrix_ad_ex11  
from gallery_section_05 import assemble_matrix_ad_ex12  

from gallery_section_05 import assemble_vector_ex12
from gallery_section_05 import assemble_vector_ex22
from gallery_section_05 import assemble_norm_ex02  

assemble11_stiffness  = compile_kernel( assemble_matrix_ad_ex11, arity=2)
assemble12_stiffness  = compile_kernel( assemble_matrix_ad_ex12, arity=2)

assemble12_rhs        = compile_kernel(assemble_vector_ex12, arity=1)
assemble22_rhs        = compile_kernel(assemble_vector_ex22, arity=1)
assemble2_norm_l2     = compile_kernel(assemble_norm_ex02, arity=1)

#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
#..
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   tabulate                     import tabulate
import numpy                        as     np
import argparse

#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists	

#===================================================================
#====   = =======   Elasticity ALGORITHM === ==      == ============
#===================================================================
def Elasticity_solve(V1, V2 , V, En, nu, Tx, u11_mae = None, u12_mae = None):
    
    mu              = En/(2.*(1+nu))
    lanbda_3d       = nu*En/((1+nu)*(1-2.*nu))
    lanbda          = 2.*lanbda_3d*mu/(lanbda_3d+2.*mu)
    # ...
    n_basis         = V1.nbasis*(V2.nbasis-1+1)
    V_basis         = (V1.nbasis,V2.nbasis-1+1)
    u1              = StencilVector(V.vector_space)
    u2              = StencilVector(V.vector_space)
    # ...
    stiffness11     = assemble11_stiffness(V, fields=[u11_mae, u12_mae], value = [ mu, (2.*mu+lanbda)])
    # stiffness11     = (stiffness11.tosparse()).toarray()
    # stiffness11     = stiffness11.reshape((V1.nbasis,V2.nbasis,V1.nbasis,V2.nbasis))
    # stiffness11     = stiffness11[:,:-1,:,:-1]
    # stiffness11     = stiffness11.reshape((n_basis,n_basis))
    stiffness11     = apply_dirichlet(V, stiffness11, dirichlet = [[False,False], [False,True]])    

    stiffness12     = assemble12_stiffness(V, fields=[u11_mae, u12_mae], value = [ mu, lanbda])
    # stiffness12     = (stiffness12.tosparse()).toarray()
    # stiffness12     = stiffness12.reshape((V1.nbasis,V2.nbasis,V1.nbasis,V2.nbasis))
    # stiffness12     = stiffness12[:,:-1,:,:-1]
    # stiffness12     = stiffness12.reshape((n_basis,n_basis))
    stiffness12     = apply_zeros(V, stiffness12, app_zeros = [[False,False], [False,True]])


    stiffness21     = assemble12_stiffness(V, fields=[u11_mae, u12_mae], value = [ lanbda, mu])
    # stiffness21     = (stiffness21.tosparse()).toarray()
    # stiffness21     = stiffness21.reshape((V1.nbasis,V2.nbasis,V1.nbasis,V2.nbasis))
    # stiffness21     = stiffness21[:,1:,:,1:]
    # stiffness21     = stiffness21.reshape((n_basis,n_basis))
    stiffness21     = apply_zeros(V, stiffness21, app_zeros = [[False,False], [True,False]])

    stiffness22     = assemble11_stiffness(V, fields=[u11_mae, u12_mae], value = [ (2.*mu+lanbda), mu])
    # stiffness22     = (stiffness22.tosparse()).toarray()
    # stiffness22     = stiffness22.reshape((V1.nbasis,V2.nbasis,V1.nbasis,V2.nbasis))
    # stiffness22     = stiffness22[:,1:,:,1:]
    # stiffness22     = stiffness22.reshape((n_basis,n_basis))
    stiffness22     = apply_dirichlet(V, stiffness22, dirichlet = [[False,False], [True,False]])

    #...
    rhs1            = assemble12_rhs( V, fields=[u11_mae, u12_mae], value = [Tx])
    # rhs1            = rhs1.toarray()
    # rhs1            = rhs1.reshape((V.nbasis))
    # rhs1            = rhs1[:,:-1]
    # rhs1            = rhs1.reshape(n_basis)
    rhs1            = apply_dirichlet(V, rhs1, dirichlet = [[False,False], [False,True]])


    rhs2            = assemble22_rhs( V, fields=[u11_mae, u12_mae], value = [Tx])
    # rhs2            = rhs2.toarray()
    # rhs2            = rhs2.reshape(V.nbasis)
    # rhs2            = rhs2[:,1:]
    # rhs2            = rhs2.reshape(n_basis)
    rhs2            = apply_dirichlet(V, rhs2, dirichlet = [[False,False], [True,False]])

    #  ---- Assembles a global linear system
    M                          = zeros((n_basis*2,n_basis*2))
    # M[:n_basis,:n_basis]       = stiffness11
    # M[:n_basis,n_basis:]       = stiffness12
    # # ...
    # M[n_basis:,:n_basis]       = stiffness21
    # M[n_basis:,n_basis:]       = stiffness22
    M[:n_basis,:n_basis]       = (stiffness11.tosparse()).toarray()[:,:]
    M[:n_basis,n_basis:]       = (stiffness12.tosparse()).toarray()[:,:]
    # ...
    M[n_basis:,:n_basis]       = (stiffness21.tosparse()).toarray()[:,:]
    M[n_basis:,n_basis:]       = (stiffness22.tosparse()).toarray()[:,:]
    # ..
    b                          = zeros(n_basis*2)
    # b[:n_basis]                = rhs1[:] 
    # b[n_basis:]                = rhs2[:]
    b[:n_basis]                = rhs1.toarray()[:] 
    b[n_basis:]                = rhs2.toarray()[:]

    #------Solve a linear system
    cond_M        = linalg.cond(M)
    if cond_M < 1e8 :
        solver_n     = "GMRES"
        #--Solve a linear system
        M               =  csc_matrix(M)
        #+++ 
        x, inf          = sla.gmres(M, b, tol=1.e-10, maxiter=5000)
    else :
        solver_n     = "conjugate gradient"
        x, inf       = sla.cg(csc_matrix(M),b, tol=1e-10)

    # ------
    x1             = zeros(V.nbasis)
    x1[:,:]      = (x[:n_basis]).reshape(V_basis)
    u1.from_array(V, x1)
    #u1            = apply_dirichlet(V, u1, dirichlet = [[False,False], [True,False]])    
    #x1            = (u1.toarray()).reshape(V.nbasis)

    x2             = zeros(V.nbasis)    
    x2[:,:]       = (x[n_basis:]).reshape(V_basis)
    u2.from_array(V, x2)
    #u2            = apply_dirichlet(V, u2, dirichlet = [[False,False], [False,True]])    
    #x2            = (u2.toarray()).reshape(V.nbasis)
    # -----
    Norm          = assemble2_norm_l2(V, fields=[u11_mae, u12_mae, u1, u2], value = [ mu, lanbda, Tx]) 
    norm          = Norm.toarray()[0]

    # ------
    #print("-------------------------------------------- error = {}-------------------cond_M = {}----------------- solver  ={}\n\n".format( norm[2], cond_M, solver_n))
    return x1, x2, norm
	
# Argument parser
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
args = parser.parse_args()
# ___
nbpts           = 100
# ___ data for Elasiticity
En              = 1e5
nu              = 0.3
Tx              = 10.
mu              = En/(2.*(1+nu))
lanbda_3d       = nu*En/((1+nu)*(1-2.*nu))
lanbda          = 2.*lanbda_3d*mu/(lanbda_3d+2.*mu)

# # ........................................................
# ....................For generating tables
# #.........................................................
degree          = 2
order_cv        = 0.
quad_degree     =  degree+3

# ... new discretization for plot
#+++++++++++++++++++++++++++++++++++++++++++++++++++++
# ... Initialisation of Picard algorithm in coarse grids
# ... Initialisation and computing optimal mapping for 16*16
# create the spline space for each direction ***MAE***
VH1 = SplineSpace(degree=degree, nelements= 16, nderiv = 2, quad_degree = quad_degree)
VH2 = SplineSpace(degree=degree, nelements= 16, nderiv = 2, quad_degree = quad_degree)
# create the tensor space
VH = TensorSpace(VH1, VH2)

#... computes optimal mapping
start        = time.time()
mp           = getGeometryMap('../fields/elasticity.xml', 0)
x11uh, x12uh = mp.coefs()

print("	\subcaption{Degree $p =",degree,"$}")
print("	\\begin{tabular}{r c c c c c}")
print("		\hline")
print("		$\#$cells &  CPU-time (s) & $l^2$-err-r & order &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
print("		\hline")
for nb_ne in range(4,6):
    
    nelements       =  2**nb_ne
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++
    # ... Initialisation of Picard algorithm in fine grid
    Vh1          = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
    Vh2          = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
    # create the tensor space
    Vh           = TensorSpace(Vh1, Vh2)
    u11_pH       = StencilVector(Vh.vector_space)
    u12_pH       = StencilVector(Vh.vector_space)

    M_prolongate = prolongation_matrix(VH, Vh)
    #... computes optimal mapping
    start        = time.time()
    mp           = getGeometryMap('../fields/elasticity.xml', 0)
    x11uh, x12uh = mp.coefs()
    
    # ...
    x11uh        = M_prolongate.dot(x11uh.reshape(VH1.nbasis*VH2.nbasis)).reshape(Vh.nbasis)
    x12uh        = M_prolongate.dot(x12uh.reshape(VH1.nbasis*VH2.nbasis)).reshape(Vh.nbasis)
    u11_pH.from_array(Vh, x11uh)
    u12_pH.from_array(Vh, x12uh)

    start = time.time()
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    xuhEL_1, xuhEL_2, error_l2 = Elasticity_solve(Vh1, Vh2, Vh, En, nu, Tx, u11_mae = u11_pH, u12_mae = u12_pH)
    spu_time = time.time() - start
    #error_l2, spu_time, xuhEL_1, xuhEL_2, x11uh, x12uh = Elasticity(nb_ne, degree = degree, Eldegree = degree)

    nbpts           = 2**nb_ne+2
    #--- Compute a solution
    ux, uxx, uxy  = pyccel_sol_field_2d((nbpts,nbpts), x11uh, Vh.knots, Vh.degree)[0:3]
    uy, uyx, uyy  = pyccel_sol_field_2d((nbpts,nbpts), x12uh, Vh.knots, Vh.degree)[0:3]
    det = uxx*uyy-uxy*uyx

    # ...
    det_min          = np.min(det)
    det_max          = np.max(det)
    #print(det_min, det_max)
    #---Compute a solution	
    uEl_1, sx_1, sy_1 = pyccel_sol_field_2d((nbpts,nbpts), xuhEL_1, Vh.knots, Vh.degree)[0:3]
    uEl_2, sx_2, sy_2 = pyccel_sol_field_2d((nbpts,nbpts), xuhEL_2, Vh.knots, Vh.degree)[0:3]  
    # ----
    uElx_1   = (uyy*sx_1 - uyx*sy_1)
    uEly_1   = (uxx*sy_1 - uxy*sx_1)
    
    uElx_2   = (uyy*sx_2 - uyx*sy_2)
    uEly_2   = (uxx*sy_2 - uxy*sx_2)
    sigma_xx = ( (2.*mu+lanbda) * uElx_1 + lanbda * uEly_2 )/det

    # ... scientific format
    MG_time          = round(spu_time, 3)
    l2_err           = np.format_float_scientific(error_l2, unique=False, precision=3)
    det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
    det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
    
    if nb_ne >4 :
        order_cv         = (np.log(error_l2)-log_b)/(np.log(1/(2**nb_ne)) - np.log(1/(2**(nb_ne-1))))
        order_cv         = round(order_cv, 3)
    log_b = np.log(error_l2)
    
    print("		",2**nb_ne, "&",  MG_time, "&", l2_err,"&", order_cv,"&", det_min,"&", det_max, "\\\\")
print("		\hline")
print("	\end{tabular}")
print('\n')

#---------------------------------------------------------
fig =plt.figure() 
for i in range(nbpts):
   phidx = ux[:,i]
   phidy = uy[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
for i in range(nbpts):
   phidx = ux[i,:]
   phidy = uy[i,:]

   plt.plot(phidx, phidy, '-b', linewidth = .3)
#plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
phidx = ux[:,0]
phidy = uy[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
# ...
phidx = ux[:,nbpts-1]
phidy = uy[:,nbpts-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
#''
phidx = ux[0,:]
phidy = uy[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
# ...
phidx = ux[nbpts-1,:]
phidy = uy[nbpts-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

#plt.xlim([-0.075,0.1])
#plt.ylim([-0.25,-0.1])
plt.axis('off')
plt.margins(0,0)
fig.tight_layout()
plt.savefig('figs/meshes.png')

fig, axes =plt.subplots() 
im2 = plt.contourf( ux, uy, sigma_xx, np.linspace(np.min(sigma_xx),np.max(sigma_xx),100), cmap= 'jet')
divider = make_axes_locatable(axes) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im2, cax=cax) 
fig.tight_layout()
plt.savefig('figs/stress_x.png')

fig, axes =plt.subplots() 
im2 = plt.contourf( ux, uy, uEl_1, np.linspace(np.min(uEl_1),np.max(uEl_1),100), cmap= 'jet')
divider = make_axes_locatable(axes) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im2, cax=cax) 
fig.tight_layout()
plt.savefig('figs/displacemenet_x.png')


fig, axes =plt.subplots() 
im2 = plt.contourf( ux, uy, uEl_2, np.linspace(np.min(uEl_2),np.max(uEl_2),100), cmap= 'jet')
divider = make_axes_locatable(axes) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
plt.colorbar(im2, cax=cax) 
fig.tight_layout()
plt.savefig('figs/displacemenet_y.png')



if args.plot :
    plt.show(block=True)
    plt.close()
else :
    plt.show(block=False)
    plt.close()
    print("Plotting is disabled. No files were saved, re-run with --plot or go to /figs")