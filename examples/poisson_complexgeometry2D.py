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

# ... Poisson tools in uniform mesh
from examples.gallery.gallery_section_06             import assemble_matrix_un_ex01
from examples.gallery.gallery_section_06             import assemble_vector_un_ex01
from examples.gallery.gallery_section_06             import assemble_norm_un_ex01

assemble_matrix_un   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

#..
from   scipy.sparse                 import csr_matrix
from   scipy.sparse                 import csc_matrix, linalg as sla
from   numpy                        import zeros, linalg, asarray
from   numpy                        import cos, sin, pi, exp, sqrt, arctan2
from   numpy                        import cosh, sinh
from   tabulate                     import tabulate
import numpy                        as     np
#...
import timeit
import time
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#==============================================================================
#.......Poisson ALGORITHM
#==============================================================================
def poisson_solve(V, u11_mpH, u12_mpH, u_d):

    # start_pois =  time.time()
    u   = StencilVector(V.vector_space)
    stiffness  = assemble_matrix_un(V, fields=[u11_mpH, u12_mpH])
    stiffness  = apply_dirichlet(V, stiffness)
    M          = stiffness.tosparse()
    # print("assmebling: ", time.time() - start_pois)
    # start_pois =  time.time()
    rhs        = assemble_rhs_un( V, fields=[u11_mpH, u12_mpH, u_d])
    rhs        = apply_dirichlet(V, rhs)
    b          = rhs.toarray()
    # print("rhs assembing: ", time.time() - start_pois)
    # start_pois =  time.time()
    #--Solve a linear system
    lu         = sla.splu(csc_matrix(M))
    x          = lu.solve(b)
    # print("solve: ", time.time() - start_pois)
    # ... assemble dirichlet boundary condition
    x          = x.reshape(V.nbasis)
    x         += (u_d.toarray()).reshape(V.nbasis)
    u.from_array(V, x)

    #--Computes error l2 and H1
    Norm    = assemble_norm_un(V, fields=[u11_mpH, u12_mpH, u])
    norm    = Norm.toarray()
    l2_norm = norm[0]
    H1_norm = norm[1]
    return u, x, l2_norm, H1_norm


nbpts       = 100 # FOR PLOT
RefinNumber = 2 # for refinement
nelements   = 16
table       = zeros((RefinNumber+1,5))
i           = 1
times       = []

#--------------------------------------------------------------
#..... Exact slution if offered or Dirichlet boundary condition
#--------------------------------------------------------------
#.. test 0
# u_exact   = lambda x, y : sin(2.*pi*x)*sin(2.*pi*y)
# g         = ['sin(2.*pi*x)*sin(2.*pi*y)']
#.. test 1
u_exact   = lambda x, y : 1./(1.+exp((x + y  - 0.5)/0.01) )
g         = ['1./(1.+exp((x + y  - 0.5)/0.01) )']

#----------------------------------------
#.....Extract CAD geometry
#----------------------------------------
#geometry = '../fields/quart_annulus.xml'
#geometry = '../fields/unitSquare.xml'
geometry = '../fields/circle.xml'
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)

# ... Assembling mapping
mp             = getGeometryMap(geometry,0)
degree         = mp.degree
quad_degree    = max(degree[0],degree[1])+3
mp.nurbs_check = True # activate NURBS if the geometry is defined by NURBS

# ... Prolongation by knots insertion matrix of the initial mapping
Nelements=(nelements,nelements)
weight, xmp, ymp       = mp.RefineGeometryMap(Nelements=Nelements)
wm1, wm2 = weight[:,0], weight[0,:]    

#--------------------------------------------------------------
#..... Initialisation and computing optimal mapping for 16*16
#--------------------------------------------------------------
# create the spline space for each direction
V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,Nelements), nderiv = 1, omega = wm1, quad_degree = quad_degree)
V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,Nelements), nderiv = 1, omega = wm2, quad_degree = quad_degree)
# create the tensor space
Vh00 = TensorSpace(V1, V2)

# ...
u11_mpH        = StencilVector(Vh00.vector_space)
u12_mpH        = StencilVector(Vh00.vector_space)
u11_mpH.from_array(Vh00, xmp)
u12_mpH.from_array(Vh00, ymp)
#--------------------------------------------------------------
# ...Assembling Dirichlet boundary condition
#--------------------------------------------------------------
print("(#=assembled Dirichlet, #=solve poisson)\n")
u_d = build_dirichlet(Vh00, g, map = (xmp, ymp))[1]
print('#')

#...solve poisson
start = time.time()
u_pH, xuh, l2_error, H1_error = poisson_solve(Vh00, u11_mpH, u12_mpH, u_d)
times.append(time.time()- start)
xuh_uni = xuh
print('#')

# ... 
table[0,:] = [degree[0], nelements, l2_error, H1_error, times[-1]]
#..for cpu time
i_save = 1
for nbne in range(RefinNumber):
    nelements = 2**(5+nbne)
    Nelements = (nelements,nelements)
    print('#---IN-UNIFORM--MESH', nelements)
    #---------------------------------------------------------------
    #.. Prologation by knots insertion matrix of the initial mapping
    #---------------------------------------------------------------
    weight, xmp, ymp       = mp.RefineGeometryMap(Nelements=Nelements)
    wm1, wm2 = weight[:,0], weight[0,:] 
    print('#refine geometry')
    V1 = SplineSpace(degree=degree[0], grid = mp.Refinegrid(0,Nelements), nderiv = 1, omega = wm1, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree[1], grid = mp.Refinegrid(1,Nelements), nderiv = 1, omega = wm2, quad_degree = quad_degree)
    # create the tensor space
    Vh00 = TensorSpace(V1, V2)
    print('#spces')
    # ...
    u11_mph         = StencilVector(Vh00.vector_space)
    u12_mph         = StencilVector(Vh00.vector_space)
    u11_mph.from_array(Vh00, xmp)
    u12_mph.from_array(Vh00, ymp)	
    #-----------------------------------------
    u_d   = build_dirichlet(Vh00, g, map = (xmp, ymp))[1]
    print('#')
    # ...
    start = time.time()
    u, xuh, l2_error,  H1_error         = poisson_solve(Vh00, u11_mph, u12_mph, u_d)
    times.append(time.time()- start)
    print('#')
    # ... Update table and iteration
    table[i_save,:]                     = [degree[0], nelements, l2_error, H1_error, times[-1]]
    i_save                             += 1

#---print errror results
#~~~~~~~~~~~~
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

# -----------------------------END OF THE SHARED PART FOR ALL GEOMETRY
from simplines    import paraview_nurbsSolutionMultipatch
paraview_nurbsSolutionMultipatch(nbpts, [Vh00], [xmp], [ymp],  xuh = [xuh], Func = u_exact)

"""
#from matplotlib.pyplot import plot, show
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   mpl_toolkits.mplot3d         import axes3d
from   matplotlib                   import cm
from   mpl_toolkits.mplot3d.axes3d  import get_test_data
from   matplotlib.ticker            import LinearLocator, FormatStrFormatter
font = {'family': 'serif', 
         'color':  'k', 
         'weight': 'normal', 
         'size': 15, 
         } 
#---Solution in uniform mesh
u, ux, uy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  xuh , Vh00.knots, Vh00.degree)
#.. circle 
#---Compute a mapping
F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp, Vh00.knots, Vh00.degree)[0]
F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp, Vh00.knots, Vh00.degree)[0]
#===============
# ... test 2
Sol_un = u_exact( F1 , F2)
fig, axes =plt.subplots() 
levelsc_un= np.linspace(np.min(u), np.max(u), 100)
im2 = plt.contourf( F1, F2, u, levelsc_un, cmap= 'jet')
divider = make_axes_locatable(axes) 
cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
cbar = plt.colorbar(im2, cax=cax) 
cbar.ax.tick_params(labelsize=15) 
cbar.ax.yaxis.label.set_fontweight('bold')
# Set axes (ticks) font weight
for label in axes.get_xticklabels() + axes.get_yticklabels():
    label.set_fontweight('bold') 
fig.tight_layout()
plt.savefig('figs/solution.png')
plt.show(block=False)
plt.close()
print("Plotting is disabled. No files were saved, run in your terminal open ./figs/solution.png")
"""