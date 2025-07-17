"""
poisson_complexgeometry2D.py

Example: Solving Poisson's Equation on a 2D complex geometry using B-spline or NURBS representation.

Author: M. Bahari
"""
from   simplines                    import compile_kernel
from   simplines                    import apply_dirichlet

from   simplines                    import SplineSpace
from   simplines                    import TensorSpace
from   simplines                    import StencilMatrix
from   simplines                    import StencilVector

from   simplines                    import pyccel_sol_field_2d
from   simplines                    import prolongation_matrix
from   simplines                    import least_square_Bspline
from   simplines                    import getGeometryMap
from   simplines                    import build_dirichlet

# Import Poisson assembly tools for uniform mesh
from examples.gallery.gallery_section_06             import assemble_matrix_un_ex01
from examples.gallery.gallery_section_06             import assemble_vector_un_ex01
from examples.gallery.gallery_section_06             import assemble_norm_un_ex01

assemble_matrix_un   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   numpy                        import cosh, sinh
from   tabulate                     import tabulate
import numpy                        as     np
import timeit
import time
import argparse

#------------------------------------------------------------------------------
# Create directory for figures if it doesn't exist
#------------------------------------------------------------------------------
import os
os.makedirs("figs", exist_ok=True)

#------------------------------------------------------------------------------
# Poisson solver algorithm
#------------------------------------------------------------------------------
def poisson_solve(V, u11_mph, u12_mph, u_d):
    u   = StencilVector(V.vector_space)
    # Assemble stiffness matrix
    stiffness  = assemble_matrix_un(V, fields=[u11_mph, u12_mph])
    stiffness  = apply_dirichlet(V, stiffness)
    M          = stiffness.tosparse()

    # Assemble right-hand side vector
    rhs        = assemble_rhs_un( V, fields=[u11_mph, u12_mph, u_d])
    rhs        = apply_dirichlet(V, rhs)
    b          = rhs.toarray()
    
    # Solve linear system
    lu         = sla.splu(csc_matrix(M))
    x          = lu.solve(b)

    # Apply Dirichlet boundary conditions
    x          = x.reshape(V.nbasis)
    x         += (u_d.toarray()).reshape(V.nbasis)
    u.from_array(V, x)

    # Compute L2 and H1 errors
    Norm    = assemble_norm_un(V, fields=[u11_mph, u12_mph, u])
    norm    = Norm.toarray()
    l2_norm = norm[0]
    H1_norm = norm[1]
    return u, x, l2_norm, H1_norm
#------------------------------------------------------------------------------
# Argument parser for controlling plotting
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
args = parser.parse_args()

#------------------------------------------------------------------------------
# Parameters and initialization
#------------------------------------------------------------------------------
nbpts       = 100 # Number of points for plotting
RefinNumber = 2   # Number of global mesh refinements
nelements   = 16  # Initial mesh size
table       = zeros((RefinNumber+1,5))
i           = 1
times       = []

print("(#=assembled Dirichlet, #=solve poisson)\n")

#------------------------------------------------------------------------------
# Define exact solution and Dirichlet boundary condition
#------------------------------------------------------------------------------
# Test 0
# u_exact   = lambda x, y : sin(2.*pi*x)*sin(2.*pi*y)
# g         = ['sin(2.*pi*x)*sin(2.*pi*y)']
# Test 1
u_exact   = lambda x, y : 1./(1.+exp((x + y  - 0.5)/0.01) )
g         = ['1./(1.+exp((x + y  - 0.5)/0.01) )']

#------------------------------------------------------------------------------
# Load CAD geometry
#------------------------------------------------------------------------------
#geometry = '../fields/quart_annulus.xml'
#geometry = '../fields/unitSquare.xml'
geometry = '../fields/circle.xml'
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)
print("Dirichlet boundary conditions", g)

# Extract geometry mapping
mp             = getGeometryMap(geometry,0)
degree         = mp.degree # Use same degree as geometry
quad_degree    = max(degree[0],degree[1])+3 # Quadrature degree
mp.nurbs_check = True # Activate NURBS if geometry uses NURBS

#------------------------------------------------------------------------------
# Initialize spaces and mapping for initial mesh
#------------------------------------------------------------------------------
Nelements        = (nelements,nelements)
weight, xmp, ymp = mp.RefineGeometryMap(Nelements=Nelements)
wm1, wm2         = weight[:,0], weight[0,:]    

# Create spline spaces for each direction
V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,Nelements), nderiv = 1, omega = wm1, quad_degree = quad_degree)
V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,Nelements), nderiv = 1, omega = wm2, quad_degree = quad_degree)
# Create tensor product space
Vh = TensorSpace(V1, V2)

# Initialize mapping vectors
u11_mph        = StencilVector(Vh.vector_space)
u12_mph        = StencilVector(Vh.vector_space)
u11_mph.from_array(Vh, xmp)
u12_mph.from_array(Vh, ymp)

#------------------------------------------------------------------------------
# Assemble Dirichlet boundary conditions
#------------------------------------------------------------------------------
u_d = build_dirichlet(Vh, g, map = (xmp, ymp))[1]
print('#')

# Solve Poisson equation on coarse grid
start = time.time()
u_pH, xuh, l2_error, H1_error = poisson_solve(Vh, u11_mph, u12_mph, u_d)
times.append(time.time()- start)
xuh_uni = xuh
print('#')

# Store results in table
table[0,:] = [degree[0], nelements, l2_error, H1_error, times[-1]]

#------------------------------------------------------------------------------
# Mesh refinement loop
#------------------------------------------------------------------------------
i_save = 1
for nbne in range(RefinNumber):
    # Refine mesh
    nelements = 2**(5+nbne)
    Nelements = (nelements,nelements)
    print('#---IN-UNIFORM--MESH', nelements)
    # Refine geometry mapping
    weight, xmp, ymp  = mp.RefineGeometryMap(Nelements=Nelements)
    wm1, wm2 = weight[:,0], weight[0,:] 
    # Create spline spaces for refined mesh
    V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,Nelements), nderiv = 1, omega = wm1, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,Nelements), nderiv = 1, omega = wm2, quad_degree = quad_degree)
    Vh = TensorSpace(V1, V2)
    print('#spces')
    # Update mapping vectors
    u11_mph         = StencilVector(Vh.vector_space)
    u12_mph         = StencilVector(Vh.vector_space)
    u11_mph.from_array(Vh, xmp)
    u12_mph.from_array(Vh, ymp)	
    # Assemble Dirichlet boundary conditions
    u_d   = build_dirichlet(Vh, g, map = (xmp, ymp))[1]
    print('#')
    # Solve Poisson equation on refined mesh
    start = time.time()
    u, xuh, l2_error,  H1_error         = poisson_solve(Vh, u11_mph, u12_mph, u_d)
    times.append(time.time()- start)
    print('#')
    # Store results
    table[i_save,:]                     = [degree[0], nelements, l2_error, H1_error, times[-1]]
    i_save                             += 1

#------------------------------------------------------------------------------
# Print error results in LaTeX table format
#------------------------------------------------------------------------------
if True :
    print("	\subcaption{Degree $p =",degree,"$}")
    print("	\\begin{tabular}{c|ccc|ccc}")
    print("		\hline")
    print("		 $\#$cells & $L^2$-Err & $H^1$-Err & cpu-time\\\\")
    print("		\hline")
    for i in range(0,RefinNumber+1):
        print("		",int(table[i,1]),"$\\times$", int(table[i,1]), "&", np.format_float_scientific(table[i,2], unique=False, precision=2), "&", np.format_float_scientific(table[i,3], unique=False, precision=2), "&", np.format_float_scientific(table[i,4], unique=False, precision=2),"\\\\")
    print("		\hline")
    print("	\end{tabular}")
print('\n')

#------------------------------------------------------------------------------
# Export solution for visualization
#------------------------------------------------------------------------------
from simplines    import paraview_nurbsSolutionMultipatch
solutions = [
    {"name": "Solution", "data": [xuh]}
]
paraview_nurbsSolutionMultipatch(nbpts, [Vh], [xmp], [ymp],  solution = solutions, Func = u_exact)
#------------------------------------------------------------------------------
# Show or close plots depending on argument
if args.plot :
    import subprocess

    # Load the multipatch VTM
    subprocess.run(["paraview", "figs/multipatch_solution.vtm"])