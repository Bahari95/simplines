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

# ... Poisson tools in uniform mesh
from gallery_section_06             import assemble_matrix_un_ex01
from gallery_section_06             import assemble_vector_un_ex01
from gallery_section_06             import assemble_norm_un_ex01

assemble_matrix_un   = compile_kernel(assemble_matrix_un_ex01, arity=2)
assemble_rhs_un      = compile_kernel(assemble_vector_un_ex01, arity=1)
assemble_norm_un     = compile_kernel(assemble_norm_un_ex01, arity=1)

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
def poisson_solve(V1, V2, V, u11_mpH, u12_mpH, u_d):

    u   = StencilVector(V.vector_space)
    stiffness = assemble_matrix_un(V, fields=[u11_mpH, u12_mpH])
    stiffness = apply_dirichlet(V, stiffness)
    rhs       = assemble_rhs_un( V, fields=[u11_mpH, u12_mpH, u_d])
    rhs       = apply_dirichlet(V, rhs)

    #--Solve a linear system
    M = stiffness.tosparse()

    cond_M = linalg.cond(M.toarray())

    lu = sla.splu(csc_matrix(M))

    b  = rhs.toarray()

    x  = lu.solve(b)
    # ... assemble dirichlet boundary condition
    x  = x.reshape(V.nbasis)
    x += (u_d.toarray()).reshape(V.nbasis)
    u.from_array(V, x)

    #--Computes error l2 and H1
    Norm    = assemble_norm_un(V, fields=[u11_mpH, u12_mpH, u])
    norm    = Norm.toarray()
    l2_norm = norm[0]
    H1_norm =  norm[1]
    return u, x, l2_norm, H1_norm, cond_M


degree      = 2
quad_degree = degree+3
nbpts       = 100 # FOR PLOT
RefinNumber = 1 # for refinement
table       = zeros((RefinNumber+1,6))
i           = 1
times       = []

#--------------------------------------------------------------
#..... Exact slution if offered or Dirichlet boundary condition
#--------------------------------------------------------------
#.. test 0
u_exact   = lambda x, y : sin(2.*pi*x)*sin(2.*pi*y)
# ... function at each part of the boundary

#--------------------------------------------------------------
#..... Initialisation and computing optimal mapping for 16*16
#--------------------------------------------------------------
nelements  = 16
# create the spline space for each direction
VH1        = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
VH2        = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
VH00       = TensorSpace(VH1, VH2)

#----------------------------------------
#..... Parameterization from 16*16 elements
#----------------------------------------
# ... Circle
geometry = '../fields/DDM.xml'
print('#---IN-UNIFORM--MESH-Poisson equation', geometry)

# ... Assembling mapping
mp             = getGeometryMap(geometry,0)
xmp = zeros(VH00.nbasis)
ymp = zeros(VH00.nbasis)

xmp[:,:], ymp[:,:]       = mp.coefs()

# ...
u11_mpH        = StencilVector(VH00.vector_space)
u12_mpH        = StencilVector(VH00.vector_space)
u11_mpH.from_array(VH00, xmp)
u12_mpH.from_array(VH00, ymp)
#--------------------------------------------------------------
# ...Assembling Dirichlet boundary condition
#--------------------------------------------------------------
print("(#=assembled Dirichlet, #=solve poisson)\n")
n_dir   = VH1.nbasis + VH2.nbasis+100
sX      = pyccel_sol_field_2d((n_dir,n_dir),  xmp , VH00.knots, VH00.degree)[0]
sY      = pyccel_sol_field_2d((n_dir,n_dir),  ymp , VH00.knots, VH00.degree)[0]

u_d                  = StencilVector(VH00.vector_space)
x_d                  = np.zeros(VH00.nbasis)
x_d[0, : ]           = least_square_Bspline(VH2.degree, VH2.knots, u_exact(sX[0, :], sY[ 0,:]), m= n_dir)
x_d[VH1.nbasis-1, :] = least_square_Bspline(VH2.degree, VH2.knots, u_exact(sX[-1,:], sY[-1,:]), m= n_dir)
x_d[:,0]             = least_square_Bspline(VH1.degree, VH1.knots, u_exact(sX[:, 0], sY[:, 0]), m= n_dir)
x_d[:, VH2.nbasis-1] = least_square_Bspline(VH1.degree, VH1.knots, u_exact(sX[:,-1], sY[:,-1]), m= n_dir)
u_d.from_array(VH00, x_d)
print('#')

#...solve poisson
start = time.time()
u_pH, xuh, l2_error, H1_error, cond = poisson_solve(VH1, VH2, VH00, u11_mpH, u12_mpH, u_d)
times.append(time.time()- start)
xuh_uni = xuh
print('#')

# ... 
table[0,:] = [degree, nelements, l2_error, H1_error, cond, times[-1]]
#..for cpu time
i_save = 1
for nbne in range(RefinNumber):
    nelements = 2**(5+nbne)
    # create the spline space for each direction
    V1 = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
    V2 = SplineSpace(degree=degree, nelements= nelements, nderiv = 2, quad_degree = quad_degree)
    # create the tensor space
    Vh00 = TensorSpace(V1, V2)

    #---------------------------------------------------------------
    #.. Prologation by knots insertion matrix of the initial mapping
    #---------------------------------------------------------------
    M_mp            = prolongation_matrix(VH00, Vh00)
    xmp             = (M_mp.dot(u11_mpH.toarray())).reshape(Vh00.nbasis)
    ymp             = (M_mp.dot(u12_mpH.toarray())).reshape(Vh00.nbasis)
    # ...
    u11_mph         = StencilVector(Vh00.vector_space)
    u12_mph         = StencilVector(Vh00.vector_space)
    u11_mph.from_array(Vh00, xmp)
    u12_mph.from_array(Vh00, ymp)	
    #-----------------------------------------
    print('#---IN-UNIFORM--MESH', nelements)
    #-----------------------------------------
    n_dir                = V1.nbasis + V2.nbasis+100
    sX                   = pyccel_sol_field_2d((n_dir,n_dir),  xmp , Vh00.knots, Vh00.degree)[0]
    sY                   = pyccel_sol_field_2d((n_dir,n_dir),  ymp , Vh00.knots, Vh00.degree)[0]
    u_d                  = StencilVector(Vh00.vector_space)
    x_d                  = np.zeros(Vh00.nbasis)
    x_d[0, : ]           = least_square_Bspline(V2.degree, V2.knots, u_exact(sX[0, :], sY[ 0,:]), m= n_dir)
    x_d[V1.nbasis-1, :]  = least_square_Bspline(V2.degree, V2.knots, u_exact(sX[-1,:], sY[-1,:]), m= n_dir)
    x_d[:,0]             = least_square_Bspline(V1.degree, V1.knots, u_exact(sX[:, 0], sY[:, 0]), m= n_dir)
    x_d[:, V2.nbasis-1]  = least_square_Bspline(V1.degree, V1.knots, u_exact(sX[:,-1], sY[:,-1]), m= n_dir)
    u_d.from_array(Vh00, x_d)
    del sX, sY, x_d
    print('#')
    # ...
    start = time.time()
    u, xuh, l2_error,  H1_error, cond         = poisson_solve(V1, V2, Vh00, u11_mph, u12_mph, u_d)
    times.append(time.time()- start)
    print('#')
    # ... Update table and iteration
    table[i_save,:]                             = [degree, nelements, l2_error, H1_error,  cond, times[-1]]
    i_save                                     += 1

#---print errror results
#~~~~~~~~~~~~
if True :
    print("	\subcaption{Degree $p =",degree,"$}")
    print("	\\begin{tabular}{c|ccc|ccc}")
    print("		\hline")
    print("		 $\#$cells & $L^2$-Err & $H^1$-Err & cpu-time\\\\")
    print("		\hline")
    for i in range(0,RefinNumber+1):
        print("		",int(table[i,1]),"$\\times$", int(table[i,1]), "&", np.format_float_scientific(table[i,2], unique=False, precision=2), "&", np.format_float_scientific(table[i,3], unique=False, precision=2), "&", np.format_float_scientific(table[i,5], unique=False, precision=2),"\\\\")
    print("		\hline")
    print("	\end{tabular}")
print('\n')

#---Solution in uniform mesh
u, ux, uy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  xuh , Vh00.knots, Vh00.degree)
#.. circle 
#---Compute a mapping
F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp, Vh00.knots, Vh00.degree)[0]
F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp, Vh00.knots, Vh00.degree)[0]
#===============
# ... test 2
Sol_un = u_exact( F1 , F2)
# -----------------------------END OF THE SHARED PART FOR ALL GEOMETRY

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