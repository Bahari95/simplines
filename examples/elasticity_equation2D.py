"""
elasticity.py

# Elasticity example: Infinite plate with circular hole under constant in-plane tension in the x-direction

@author : M. BAHARI
"""

from simplines import compile_kernel, apply_dirichlet, apply_zeros

from simplines import SplineSpace
from simplines import TensorSpace
from simplines import StencilMatrix
from simplines import StencilVector
from simplines import sol_field_NURBS_2d
from simplines import prolongation_matrix
from simplines import getGeometryMap

import time

# Import assembly routines for elasticity equation
from examples.gallery.gallery_section_05 import assemble_matrix_ad_ex11  
from examples.gallery.gallery_section_05 import assemble_matrix_ad_ex12  
from examples.gallery.gallery_section_05 import assemble_vector_ex12
from examples.gallery.gallery_section_05 import assemble_vector_ex22
from examples.gallery.gallery_section_05 import assemble_norm_ex02  

assemble11_stiffness  = compile_kernel( assemble_matrix_ad_ex11, arity=2)
assemble12_stiffness  = compile_kernel( assemble_matrix_ad_ex12, arity=2)

assemble12_rhs        = compile_kernel(assemble_vector_ex12, arity=1)
assemble22_rhs        = compile_kernel(assemble_vector_ex22, arity=1)
assemble2_norm_l2     = compile_kernel(assemble_norm_ex02, arity=1)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data
from matplotlib.ticker import LinearLocator, FormatStrFormatter

from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix, linalg as sla
from numpy import zeros, linalg, asarray
from numpy import cos, sin, pi, exp, sqrt, arctan2
from tabulate import tabulate
import numpy as np
import argparse

#------------------------------------------------------------------------------
# Create output directory for figures
import os
os.makedirs("figs", exist_ok=True)  # Avoid error if folder already exists

#------------------------------------------------------------------------------
# Elasticity solver function
#------------------------------------------------------------------------------
def Elasticity_solve(V1, V2 , V, En, nu, Tx, u11_mae = None, u12_mae = None):
    mu              = En/(2.*(1+nu))
    lanbda_3d       = nu*En/((1+nu)*(1-2.*nu))
    lanbda          = 2.*lanbda_3d*mu/(lanbda_3d+2.*mu)

    n_basis         = V1.nbasis*(V2.nbasis-1+1)
    V_basis         = (V1.nbasis,V2.nbasis-1+1)
    u1              = StencilVector(V.vector_space)
    u2              = StencilVector(V.vector_space)

    # Assemble stiffness matrices with boundary conditions
    stiffness11     = assemble11_stiffness(V, fields=[u11_mae, u12_mae], value = [ mu, (2.*mu+lanbda)])
    stiffness11     = apply_dirichlet(V, stiffness11, dirichlet = [[False,False], [False,True]])    

    stiffness12     = assemble12_stiffness(V, fields=[u11_mae, u12_mae], value = [ mu, lanbda])
    stiffness12     = apply_zeros(V, stiffness12, app_zeros = [[False,False], [False,True]])

    stiffness21     = assemble12_stiffness(V, fields=[u11_mae, u12_mae], value = [ lanbda, mu])
    stiffness21     = apply_zeros(V, stiffness21, app_zeros = [[False,False], [True,False]])

    stiffness22     = assemble11_stiffness(V, fields=[u11_mae, u12_mae], value = [ (2.*mu+lanbda), mu])
    stiffness22     = apply_dirichlet(V, stiffness22, dirichlet = [[False,False], [True,False]])

    # Assemble right-hand side vectors with boundary conditions
    rhs1            = assemble12_rhs( V, fields=[u11_mae, u12_mae], value = [Tx])
    rhs1            = apply_dirichlet(V, rhs1, dirichlet = [[False,False], [False,True]])

    rhs2            = assemble22_rhs( V, fields=[u11_mae, u12_mae], value = [Tx])
    rhs2            = apply_dirichlet(V, rhs2, dirichlet = [[False,False], [True,False]])

    # Build global linear system
    M                          = zeros((n_basis*2,n_basis*2))
    M[:n_basis,:n_basis]       = (stiffness11.tosparse()).toarray()[:,:]
    M[:n_basis,n_basis:]       = (stiffness12.tosparse()).toarray()[:,:]
    M[n_basis:,:n_basis]       = (stiffness21.tosparse()).toarray()[:,:]
    M[n_basis:,n_basis:]       = (stiffness22.tosparse()).toarray()[:,:]

    b                          = zeros(n_basis*2)
    b[:n_basis]                = rhs1.toarray()[:] 
    b[n_basis:]                = rhs2.toarray()[:]

    # Solve the linear system using GMRES
    M               =  csc_matrix(M)
    x, inf          = sla.gmres(M, b)

    # Extract solution and convert to field representation
    x1             = zeros(V.nbasis)
    x1[:,:]        = (x[:n_basis]).reshape(V_basis)
    u1.from_array(V, x1)

    x2             = zeros(V.nbasis)    
    x2[:,:]        = (x[n_basis:]).reshape(V_basis)
    u2.from_array(V, x2)

    # Compute L2 norm of the error
    Norm           = assemble2_norm_l2(V, fields=[u11_mae, u12_mae, u1, u2], value = [ mu, lanbda, Tx]) 
    norm           = Norm.toarray()[0]

    return x1, x2, norm
    
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
args = parser.parse_args()

#------------------------------------------------------------------------------
# Problem parameters and mesh refinement
nbpts           = 100
RefinNumber     = 2

En              = 1e5
nu              = 0.3
Tx              = 10.
mu              = En/(2.*(1+nu))
lanbda_3d       = nu*En/((1+nu)*(1-2.*nu))
lanbda          = 2.*lanbda_3d*mu/(lanbda_3d+2.*mu)

order_cv        = 0. # Convergence order

Nelements      = (16, 16) # Initial number of elements

#------------------------------------------------------------------------------
# Geometry mapping and mesh refinement loop
start          = time.time()
mp             = getGeometryMap('../fields/elasticity.xml', 0)
mp.nurbs_check = True # Use NURBS mapping
degree         = mp.degree
quad_degree    = max(degree[0],degree[1])+3

print("	\subcaption{Degree $p =",degree,"$}")
print("	\\begin{tabular}{r c c c c c}")
print("		\hline")
print("		$\#$cells &  CPU-time (s) & $l^2$-err-r & order \\\\")
print("		\hline")
for nb_ne in range(4,RefinNumber+4):
    nelements       =  2**nb_ne
    Nelements       = (nelements,nelements)
    # Refine geometry and get weights and control points
    weight, xmp, ymp  = mp.RefineGeometryMap(Nelements=Nelements)
    wm1, wm2 = weight[:,0], weight[0,:] 

    V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0, Nelements), omega = wm1, nderiv = 1, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1, Nelements), omega = wm2, nderiv = 1, quad_degree = quad_degree)
    Vh = TensorSpace(V1, V2)

    u11_pH       = StencilVector(Vh.vector_space)
    u12_pH       = StencilVector(Vh.vector_space)

    u11_pH.from_array(Vh, xmp)
    u12_pH.from_array(Vh, ymp)

    start = time.time()
    # Solve elasticity problem on current mesh
    xuhEL_1, xuhEL_2, error_l2 = Elasticity_solve(V1, V2, Vh, En, nu, Tx, u11_mae = u11_pH, u12_mae = u12_pH)
    spu_time = time.time() - start

    MG_time          = round(spu_time, 3)
    l2_err           = np.format_float_scientific(error_l2, unique=False, precision=3)
    
    if nb_ne >4 :
        order_cv         = (np.log(error_l2)-log_b)/(np.log(1/(2**nb_ne)) - np.log(1/(2**(nb_ne-1))))
        order_cv         = round(order_cv, 3)
    log_b = np.log(error_l2)
    
    print("		",2**nb_ne, "&",  MG_time, "&", l2_err,"&", order_cv, "\\\\")
print("		\hline")
print("	\end{tabular}")
print('\n')

#------------------------------------------------------------------------------
# Post-processing: evaluate solution and plot results
nbpts           = 2**nb_ne+2

# Evaluate geometry mapping and its derivatives
ux, uxx, uxy  = sol_field_NURBS_2d((nbpts,nbpts), xmp, Vh.omega, Vh.knots, Vh.degree)[0:3]
uy, uyx, uyy  = sol_field_NURBS_2d((nbpts,nbpts), ymp, Vh.omega, Vh.knots, Vh.degree)[0:3]
det = uxx*uyy-uxy*uyx

det_min          = np.min(det)
det_max          = np.max(det)

# Evaluate computed displacement and stress fields
uEl_1, sx_1, sy_1 = sol_field_NURBS_2d((nbpts,nbpts), xuhEL_1, Vh.omega, Vh.knots, Vh.degree)[0:3]
uEl_2, sx_2, sy_2 = sol_field_NURBS_2d((nbpts,nbpts), xuhEL_2, Vh.omega, Vh.knots, Vh.degree)[0:3]  

uElx_1   = (uyy*sx_1 - uyx*sy_1)
uEly_1   = (uxx*sy_1 - uxy*sx_1)
uElx_2   = (uyy*sx_2 - uyx*sy_2)
uEly_2   = (uxx*sy_2 - uxy*sx_2)
sigma_xx = ( (2.*mu+lanbda) * uElx_1 + lanbda * uEly_2 )/det

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
from simplines    import paraview_nurbsSolutionMultipatch
precomputed = [
    {"name": "displacement_x", "data": [uEl_1]},
    {"name": "displacement_y", "data": [uEl_2]},
    {"name": "sigma_xx$", "data": [sigma_xx]}
]
paraview_nurbsSolutionMultipatch(nbpts, [Vh], [xmp], [ymp], precomputed = precomputed)

#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/multipatch_solution.vtm"])
else :
    print("Plotting is disabled. No files were saved, re-run with --plot or tape paraview figs/multipatch_solution.vtm in your terminal")