"""
PMAE_fast_solver.py

# Parabolic Monge-Ampere solver for fast diagonalization solver for Monge-Ampere equation

@author : M. BAHARI
"""

from   simplines                    import compile_kernel
from   simplines                    import SplineSpace
from   simplines                    import TensorSpace
from   simplines                    import StencilMatrix
from   simplines                    import StencilVector
from   simplines                    import pyccel_sol_field_2d
from   simplines                    import quadratures_in_admesh
#.. Prologation by knots insertion matrix
from   simplines                    import prolongation_matrix
# ... Using Kronecker algebra accelerated with Pyccel
from   simplines                    import Poisson
# ...   load a geometry from xml file 
from   simplines                    import getGeometryMap

from gallery_section_07             import assemble_stiffnessmatrix1D
from gallery_section_07             import assemble_massmatrix1D
from gallery_section_07             import assemble_matrix_ex01
from gallery_section_07             import assemble_matrix_ex02
#..
from gallery_section_07             import assemble_vector_ex01
from gallery_section_07             import assemble_Quality_ex01
from gallery_section_07             import assemble_vector_ex00
#...
assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex01, arity=1)
assemble_matrix_ex02 = compile_kernel(assemble_matrix_ex02, arity=1)
#..
assemble_rhs_in      = compile_kernel(assemble_vector_ex00, arity=1)
assemble_rhs         = compile_kernel(assemble_vector_ex01, arity=1)
assemble_Quality     = compile_kernel(assemble_Quality_ex01, arity=1)


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
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists

#=============================================================================================
#.......Picard ALGORITHM :  we don't use class function because the code here is just for test
#=============================================================================================
def picard_solve(V1, V2, V3, V4, V, V00, V11, V01, V10, u11_mpH = None, u12_mpH = None, times = None, x_2 = None, niter = None, tol = None):
       if niter is None :
             niter      = 100   #
       if tol is None :
          tol     = 1e-7  # 
       dt         = 0.5 # fix
       epsilon    = 1.  # fix
       gamma      = 1.  # fix
       # .. computes basis and sopans in adapted quadrature mapped by the gradient mapping
       Quad_adm   = quadratures_in_admesh(V11)
       #----------------------------------------------------------------------------------------------
       # ... Strong form of Neumann boundary condition which is Dirichlet because of Mixed formulation
       u_01       = StencilVector(V01.vector_space)
       u_10       = StencilVector(V10.vector_space)
       #..
       x_D        = np.zeros(V01.nbasis)
       y_D        = np.zeros(V10.nbasis)

       x_D[-1, :] = 1. 
       y_D[:, -1] = 1.
       #..
       u_01.from_array(V01, x_D)
       u_10.from_array(V10, y_D)

       #... We delete the first and the last spline function
       #.. as a technic for applying Neumann boundary condition
       #.in a mixed formulation

       #..Stiffness and Mass matrix in 1D in the first deriction
       D1         = assemble_mass1D(V3)
       D1         = D1.tosparse()
       D1         = D1.toarray()
       D1         = csr_matrix(D1)
       #___
       M1         = assemble_mass1D(V1)
       M1         = M1.tosparse()
       m1         = M1
       M1         = M1.toarray()[1:-1,1:-1]
       M1         = csc_matrix(M1)
       m1         = csr_matrix(m1)

       #..Stiffness and Mass matrix in 1D in the second deriction
       D2         = assemble_mass1D(V4)
       D2         = D2.tosparse()
       D2         = D2.toarray()
       D2         = csr_matrix(D2)
       #___
       M2         = assemble_mass1D(V2)
       M2         = M2.tosparse()
       m2         = M2
       M2         = M2.toarray()[1:-1,1:-1]
       M2         = csc_matrix(M2)
       m2         = csr_matrix(m2)

       #...
       R1         = assemble_matrix_ex01(V01)
       R1         = R1.toarray()
       R1         = R1.reshape(V01.nbasis)
       R1         = R1[1:-1,:]
       R1         = csr_matrix(R1)
       #___
       R2         = assemble_matrix_ex02(V10)
       R2         = R2.toarray()
       R2         = R2.reshape(V10.nbasis)
       R2         = R2[:,1:-1]
       R2         = csr_matrix(R2)

       #...step 0.1
       mats_1     = [M1, 0.5*M1]
       mats_2     = [D2, 0.5*D2]

       # ...Fast Solver
       poisson_c1 = Poisson(mats_1, mats_2)
              
       #...step 0.2
       mats_1     = [D1, 0.5*D1]
       mats_2     = [M2, 0.5*M2]

       # ...Fast Solver
       poisson_c2 = Poisson(mats_1, mats_2)
       
       #...step 2
       K1         = assemble_stiffness1D(V4)
       K1         = K1.tosparse()
       # ...
       K2         = assemble_stiffness1D(V3)
       K2         = K2.tosparse()
       mats_1     = [D1, epsilon*gamma*K1]
       mats_2     = [D2, epsilon*gamma*K2]

       # ...Fast Solver
       poisson    = Poisson(mats_1, mats_2, tau = epsilon)
       #...
       b11        = -kron(m1[1:-1,:], D2).dot(u_01.toarray())
       #___
       b12        = -kron(D1, m2[1:-1,:]).dot(u_10.toarray())

       #___
       x11_1      = kron(R1, D2)
       x12_1      = kron(D1, R2.T)
       #___
       # ... for assembling residual
       M_res      = kron(M1, D2)
       #___       
       del R1
       del R2
       del D1
       del D2
       del M1
       del M2
       # ... for Two or Multi grids
       if x_2 is None :    
           u       = StencilVector(V11.vector_space)
           # ...
           x_2     = zeros((V1.nbasis-2)*V3.nbasis)
           
           #---Assembles a right hand side of Poisson equation
           rhs          = StencilVector(V11.vector_space)
           rhs          = assemble_rhs_in(V, fields = [u11_mpH, u12_mpH], value = [epsilon, gamma, dt, times], out= rhs )
           b            = rhs.toarray()
           #___
           # ... Solve first system
           x2           = poisson.solve(b)
           #___
           u.from_array(V11, x2.reshape(V11.nbasis))
           #..Residual 
           b11 - x11_1.dot(x2)
           var_1 =  poisson_c1.solve(b11 - x11_1.dot(x2))
           dx           = var_1[:]-x_2[:]
           x_2          =  var_1[:]
       else           :       
           u            = StencilVector(V11.vector_space)
           # ...
           u.from_array(V11, x_2.reshape(V11.nbasis))
           # ...
           x_2          = zeros((V1.nbasis-2)*V3.nbasis)
       for i in range(niter):
           #---Assembles a right hand side of Poisson equation
           spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_Gradmap_quadratures(u)
           rhs          = StencilVector(V11.vector_space)
           rhs          = assemble_rhs(V, fields = [u, u11_mpH, u12_mpH], value = [epsilon, gamma, dt,  times, spans_ad1, spans_ad2, basis_ad1, basis_ad2], out= rhs )
           b            = rhs.toarray()
           # ... Solve first system
           x2           = poisson.solve(b)
           #___
           u.from_array(V11, x2.reshape(V11.nbasis))
           #..Residual 
           var_1 =  poisson_c1.solve(b11 - x11_1.dot(x2))
           dx           = var_1[:]-x_2[:]
           x_2          = var_1[:]
           #... Compute residual for L2
           l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
           if l2_residual < tol:
              break

       #___
       x_D[1:-1,:]  =  poisson_c1.solve(b11 - x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])
       u_01.from_array(V01, x_D)
       #___
       y_D[:,1:-1]  =  poisson_c2.solve(b12 - x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
       u_10.from_array(V10, y_D)
       return u_01, u_10, x_D, y_D, i, l2_residual, x2

# # .................................................................
# ....................Using Two or Multi grid method for soving MAE
# #..................................................................

def  Monge_ampere_equation(nb_ne, geometry = '../fields/circle.xml', degree = None, times = None, check =None) :
	#Degree of B-spline and number of elements
	if nb_ne <=3 :
		print('please for the reason of sufficient mesh choose nb_ne strictly greater than 3')
		return 0.
	if degree is None :
		degree          = 3
	if times is None :
			times           = 0.

	#..... Initialisation and computing optimal mapping for 16*16
	#----------------------
	# create the spline space for each direction
	Hnelements       = 2**4
	V1H             = SplineSpace(degree=degree,   nelements= Hnelements, nderiv = 2)
	V2H             = SplineSpace(degree=degree,   nelements= Hnelements, nderiv = 2)
	V3H             = SplineSpace(degree=degree-1, nelements= Hnelements, grid = V1H.grid, nderiv = 2, mixed = True)
	V4H             = SplineSpace(degree=degree-1, nelements= Hnelements, grid = V2H.grid, nderiv = 2, mixed = True)

	# create the tensor space
	VH00           = TensorSpace(V1H, V2H)
	VH11           = TensorSpace(V3H, V4H)
	VH01           = TensorSpace(V1H, V3H)
	VH10           = TensorSpace(V4H, V2H)

	# ... Assembling mapping
	V1mpH          = SplineSpace(degree=2,   nelements= Hnelements, nderiv = 2, quad_degree = degree)
	V2mpH          = SplineSpace(degree=2,   nelements= Hnelements, nderiv = 2, quad_degree = degree)	
	VHmp           = TensorSpace(V1mpH, V2mpH)

	# ...
	mp             = getGeometryMap(geometry,0)
	xmp, ymp       = mp.coefs()

	# ...

	u11_mpH        = StencilVector(VHmp.vector_space)
	u12_mpH        = StencilVector(VHmp.vector_space)
	u11_mpH.from_array(VHmp, xmp)
	u12_mpH.from_array(VHmp, ymp)
	# ... start multilevel method G-space
	VH             = TensorSpace(V3H, V4H, V1mpH, V2mpH)

	#... in coarse grid
	tol            = 1e-5
	start          = time.time()
	x2H            = picard_solve(V1H, V2H, V3H, V4H, VH, VH00, VH11, VH01, VH10, u11_mpH = u11_mpH, u12_mpH = u12_mpH, times = times, tol = tol)[-1]
	MG_time        = time.time()- start

	# ... For multigrid method
	for n in range(5,nb_ne):
		nelements   = 2**n
		V1mg        = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2)
		V2mg        = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2)
		V3mg        = SplineSpace(degree=degree-1, nelements= nelements, grid = V1mg.grid, nderiv = 2, mixed = True)
		V4mg        = SplineSpace(degree=degree-1, nelements= nelements, grid = V2mg.grid, nderiv = 2, mixed = True)

		# create the tensor space
		Vh00mg      = TensorSpace(V1mg, V2mg)
		Vh11mg      = TensorSpace(V3mg, V4mg)
		Vh01mg      = TensorSpace(V1mg, V3mg)
		Vh10mg      = TensorSpace(V4mg, V2mg)
		
		# ... Assembling mapping
		V1mph       = SplineSpace(degree=2,   nelements= nelements, nderiv = 2, quad_degree = degree)
		V2mph       = SplineSpace(degree=2,   nelements= nelements, nderiv = 2, quad_degree = degree)	
		Vhmp        = TensorSpace(V1mph, V2mph)
			
		Vhmg        = TensorSpace(V3mg, V4mg, V1mph, V2mph)

		#.. Prologation by knots insertion matrix of the initial mapping
		M_mp        = prolongation_matrix(VHmp, Vhmp)
		xmp         = (M_mp.dot(u11_mpH.toarray())).reshape(Vhmp.nbasis)
		ymp         = (M_mp.dot(u12_mpH.toarray())).reshape(Vhmp.nbasis)
		# ...
		u11_mph     = StencilVector(Vhmp.vector_space)
		u12_mph     = StencilVector(Vhmp.vector_space)
		u11_mph.from_array(Vhmp, xmp)
		u12_mph.from_array(Vhmp, ymp)	   
		
		#.. Prologation by knots insertion matrix
		M           = prolongation_matrix(VH11, Vh11mg)
		x2H         = M.dot(x2H)
		# ...

		# ... in new grid
		#tol       *= 1e-1
		start       = time.time()
		x2H         = picard_solve(V1mg, V2mg, V3mg, V4mg, Vhmg, Vh00mg, Vh11mg, Vh01mg, Vh10mg, u11_mpH = u11_mph, u12_mpH = u12_mph, times = times, x_2 = x2H, tol= tol)[-1]
		MG_time    += time.time()- start
		# .. update grids
		V1H         = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2)
		V2H         = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2)
		V3H         = SplineSpace(degree=degree-1, nelements= nelements, grid = V1H.grid, nderiv = 2, mixed = True)
		V4H         = SplineSpace(degree=degree-1, nelements= nelements, grid = V2H.grid, nderiv = 2, mixed = True)

		# create the tensor space
		VH00        = TensorSpace(V1H, V2H)
		VH11        = TensorSpace(V3H, V4H)
		VH01        = TensorSpace(V1H, V3H)
		VH10        = TensorSpace(V4H, V2H)
		VH          = TensorSpace(V1H, V2H, V3H, V4H, V1mph, V2mph )

	# ...
	if check is not None :
		if  VH.nelements[0] == Hnelements :
			print(".../!\.. : two-level is activated")
		else : 
			print(".../!\.. : multi-level is activated")
	#----------------------
	# create the spline space for each direction
	nelements       = 2**nb_ne
	V1              = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2)
	V2              = SplineSpace(degree=degree,   nelements= nelements, nderiv = 2)
	V3              = SplineSpace(degree=degree-1, nelements= nelements, grid = V1.grid, nderiv = 2, mixed = True)
	V4              = SplineSpace(degree=degree-1, nelements= nelements, grid = V2.grid, nderiv = 2, mixed = True)

	# create the tensor space
	Vh00            = TensorSpace(V1, V2)
	Vh11            = TensorSpace(V3, V4)
	Vh01            = TensorSpace(V1, V3)
	Vh10            = TensorSpace(V4, V2)

	# ... Assembling mapping
	V1mph           = SplineSpace(degree=2,   nelements= nelements, nderiv = 2, quad_degree = degree)
	V2mph           = SplineSpace(degree=2,   nelements= nelements, nderiv = 2, quad_degree = degree)	
	Vhmp            = TensorSpace(V1mph, V2mph)
			
	Vh              = TensorSpace(V3, V4, V1mph, V2mph)
	#.. Prologation by knots insertion matrix of the initial mapping
	M_mp            = prolongation_matrix(VHmp, Vhmp)
	xmp             = (M_mp.dot(u11_mpH.toarray())).reshape(Vhmp.nbasis)
	ymp             = (M_mp.dot(u12_mpH.toarray())).reshape(Vhmp.nbasis)
	# ...
	u11_mph         = StencilVector(Vhmp.vector_space)
	u12_mph         = StencilVector(Vhmp.vector_space)
	u11_mph.from_array(Vhmp, xmp)
	u12_mph.from_array(Vhmp, ymp)	   	

	#.. Prologation by knots insertion matrix
	M                = prolongation_matrix(VH11, Vh11)
	x2H              = M.dot(x2H)	   	

	#.. Prologation by knots insertion matrix
	#M                = prolongation_matrix(VH11, Vh11)
	#x2H              = M.dot(x2H)	

	# ... in fine grid
	start            = time.time()
	u11_pH, u12_pH, x11uh, x12uh, iter_N, l2_residualh, x2 = picard_solve(V1, V2, V3, V4, Vh, Vh00, Vh11, Vh01, Vh10, u11_mpH = u11_mph, u12_mpH = u12_mph, x_2 = x2H, times = times)
	MG_time          = time.time()- start
	'''
	while times <5.:
		print('times =', times)
		times += 0.01
		u11_pH, u12_pH, x11uh, x12uh, iter_N, l2_residualh, x2 = picard_solve(V1, V2, V3, V4, Vh, Vh00, Vh11, Vh01, Vh10, u11_mpH = u11_mph, u12_mpH = u12_mph, x_2 =x2, niter = 1, times = times)
	'''
	# ...
	Vh               = TensorSpace(V1, V2, V3, V4, V1mph, V2mph) 
	# .. computes basis and sopans in adapted quadrature
	Quad_adm         = quadratures_in_admesh(Vh)
	spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11_pH, u12_pH)
	Quality          = StencilVector(Vh11.vector_space)
	Quality          = assemble_Quality(Vh, fields=[u11_pH, u12_pH, u11_mph, u12_mph], value = [times, spans_ad1, spans_ad2, basis_ad1, basis_ad2],  out = Quality)
	norm             = Quality.toarray()
	l2_Quality       = norm[0]
	l2_displacement  = norm[1]
	#min_det          = norm[2]
	#print(iter_N)
	return nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vhmp


# # ........................................................
# ....................For testing in one nelements
# #.........................................................
if True :

	# ... unite-squar 0.6
	#geometry = '../fields/squar.xml'
	
	# ... Circular domain
	#geometry = '../fields/circle.xml'
	
	# ... Puzzle piece
	#geometry = '../fields/Piece.xml'
	
	# ... Quartert-annulus
	#geometry = '../fields/quart_annulus.xml'
	
	# ... IP
	geometry = '../fields/IP.xml'

	# ... Butterfly
	#geometry = '../fields/butterfly.xml'
	# ... nelement = 2**nb_ne
	nb_ne           = 5
	
	nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vhmp = Monge_ampere_equation(nb_ne, geometry=geometry, check = True)

	#---Compute a solution
	nbpts              = 100
	
	#---Solution in uniform mesh
	sx, uxx, uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh01.knots, Vh01.degree)
	sy, uyx, uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh10.knots, Vh10.degree)[0:3]
	
	#ssx = 2.*X-sx
	#ssy = 2.*Y-sy

	#sx = pyccel_sol_field_2d( None,  x11uh , Vh01.knots, Vh01.degree, meshes = (ssx, ssy))[0]
	#sy = pyccel_sol_field_2d( None,  x12uh , Vh10.knots, Vh10.degree, meshes = (ssx, ssy))[0]
	
	#print('error in meshes ==', np.min(X-sx),np.min(Y-sy)) 

	#---Compute a mapping
	F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp , Vhmp.knots, Vhmp.degree)[0]
	F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp , Vhmp.knots, Vhmp.degree)[0]
	# ... in adaped mesh
	ux = pyccel_sol_field_2d( None,  xmp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
	uy = pyccel_sol_field_2d( None,  ymp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
	# ... Jacobian function of Optimal mapping
	det = uxx*uyy-uxy*uyx

	det_min          = np.min( det[1:-1,1:-1])
	det_max          = np.max( det[1:-1,1:-1])
	#... tabulate 

	print("degree = ", Vh01.degree[0])
	table            = [ [Vh01.degree[0], nelements, l2_Quality, MG_time, l2_displacement, det_min, det_max] ]
	headers          = ["degree", "$#$cells"," Err","CPU-time (s)", "Qual" ,"$min-det(PsiPsi)$", "$max_det(PsiPsi)$"]
	print(tabulate(table, headers, tablefmt="github"),'\n')

# # ........................................................
# ....................For generating tables
# #.........................................................
if False :
	degree          = 3
	# ... new discretization for plot
	nbpts           = 100
	print("	\subcaption{Degree $p =",degree,"$}")
	print("	\\begin{tabular}{r c c c c c}")
	print("		\hline")
	print("		$\#$cells & Err & CPU-time (s) & Qual &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
	print("		\hline")
	for nb_ne in range(4,8):
		
		nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vhmp = Monge_ampere_equation(nb_ne, degree = degree)

		#---Compute a solution
		sx, uxx, uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh01.knots, Vh01.degree)
		sy, uyx, uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh10.knots, Vh10.degree)[0:3]

		#---Compute a mapping
		F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp , Vhmp.knots, Vhmp.degree)[0]
		F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp , Vhmp.knots, Vhmp.degree)[0]
		# ... in adaped mesh
		ux = pyccel_sol_field_2d( None, xmp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
		uy = pyccel_sol_field_2d( None, ymp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
		# ... Jacobian function of Optimal mapping
		det = uxx*uyy-uxy**2
		# ...
		det_min          = np.min( det[1:-1,1:-1])
		det_max          = np.max( det[1:-1,1:-1])
		
		# ... scientific format
		l2_Quality       = np.format_float_scientific(l2_Quality, unique=False, precision=3)
		l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
		MG_time          = round(MG_time, 3)
		det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
		det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
		print("		",nelements, "&", l2_Quality,"&",  MG_time, "&", l2_displacement, "&", det_min, "&", det_max,"\\\\")
	print("		\hline")
	print("	\end{tabular}")
	print('\n')

# # ........................................................
# ....................For generating tables
# #.........................................................
if False :
	degree          = 3
	# ... new discretization for plot
	nbpts           = 100
	print("	\subcaption{Degree $p =",degree,"$}")
	print("	\\begin{tabular}{r c c c c c}")
	print("		\hline")
	print("		$\#$cells & Err & CPU-time (s) &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
	print("		\hline")
	for nb_ne in range(4,8):
		
		nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vhmp = Monge_ampere_equation(nb_ne, degree = degree)
		#---Compute a solution
		sx, uxx, uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh01.knots, Vh01.degree)
		sy, uyx, uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh10.knots, Vh10.degree)[0:3]

		#---Compute a mapping
		F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp , Vhmp.knots, Vhmp.degree)[0]
		F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp , Vhmp.knots, Vhmp.degree)[0]
		# ... in adaped mesh
		ux = pyccel_sol_field_2d( None, xmp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
		uy = pyccel_sol_field_2d( None, ymp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
		# ... Jacobian function of Optimal mapping
		det = uxx*uyy-uxy**2
		# ...
		det_min          = np.min( det[1:-1,1:-1])
		det_max          = np.max( det[1:-1,1:-1])
		
		# ... scientific format
		l2_Quality       = np.format_float_scientific(l2_Quality, unique=False, precision=3)
		l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
		MG_time          = round(MG_time, 3)
		det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
		det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
		print("		",nelements, "&", l2_Quality,"&",  MG_time, "&", det_min, "&", det_max,"\\\\")
	print("		\hline")
	print("	\end{tabular}")
	print('\n')
	
# # ........................................................
# ....................For generating tables with error analysis at th bundary
# #.........................................................
if False :
	degree          = 3
	# ... new discretization for plot
	nbpts           = 100
	print("	\subcaption{Degree $p =",degree,"$}")
	print("	\\begin{tabular}{r c c c c c}")
	print("		\hline")
	print("		$\#$cells & Err & CPU-time (s) & Qual &$\min~\\text{Jac}(\PsiPsi)$ &$\max ~\\text{Jac}(\PsiPsi)$\\\\")
	print("		\hline")
	for nb_ne in range(4,8):
		
		nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10 = Monge_ampere_equation(nb_ne, degree = degree)
		#---Compute a solution
		sx, uxx, uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh01.knots, Vh01.degree)
		sy, uyx, uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh10.knots, Vh10.degree)[0:3]

		#---Compute a mapping
		F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp , Vhmp.knots, Vhmp.degree)[0]
		F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp , Vhmp.knots, Vhmp.degree)[0]
		# ... in adaped mesh
		ux = pyccel_sol_field_2d( None, xmp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
		uy = pyccel_sol_field_2d( None, ymp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
		# ... Jacobian function of Optimal mapping
		det = uxx*uyy-uxy**2
		# ...
		det_min          = np.min( det[1:-1,1:-1])
		det_max          = np.max( det[1:-1,1:-1])
		error_boundary   = max(np.max(abs(sx[-1,:] -1.)), np.max(abs(sy[:,-1]-1.)), np.max(abs(sx[0,:] )), np.max(abs(sy[:,0]))  )
		# ... scientific format
		error_boundary   = np.format_float_scientific(error_boundary, unique=False, precision=3)
		l2_Quality       = np.format_float_scientific(l2_Quality, unique=False, precision=3)
		l2_displacement  = np.format_float_scientific( l2_displacement, unique=False, precision=3)
		MG_time          = round(MG_time, 3)
		det_min          = np.format_float_scientific(det_min, unique=False, precision=3)
		det_max          = np.format_float_scientific(det_max, unique=False, precision=3)
		print("		",nelements, "&", l2_Quality,"&",  MG_time, "&", l2_displacement, "&", det_min, "&", det_max, "&",error_boundary,"\\\\")
	print("		\hline")
	print("	\end{tabular}")
	print('\n')
	
#~~~~~~~~~~~~~~~~~~~~~~~
for i in range(nbpts):
  for j in range(nbpts):
     if det[i,j] < 0.:
         print('Npoints =',nbpts,'min_Jac-F in the entire domain = ', det[i,j] ,'index =', i, j)

print('..../!\...: min~max value of the Jacobian function =', np.min(det),'~', np.max(det) )

#         -++++++++++++++++++++++++++++++++++++++++++++++++++++ End of sharing part of any geometry-----------------------------------------------------------

#.. Analytic Density function unite-square 
#rho = lambda x,y : 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2)

#rho = lambda x,y :1+5*np.exp(-100*np.abs((x-0.45)**2+(y-0.4)**2-0.09))+5.*np.exp(-100.*np.abs(x**2+y**2-0.2))+5.*np.exp(-100*np.abs((x+0.45)**2 +(y-0.4)**2-0.1))+7.*np.exp(-100.*np.abs(x**2+(y+1.25)**2-0.4))
#.. Test 1  circle
#def rho(x,y):
#   return 1. + 5./np.cosh( 5.*((x-np.sqrt(3)/2)**2+(y-0.5)**2 - (np.pi/2)**2) )**2 + 5./np.cosh( 5.*((x+np.sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2

#... QUarter annulus
rho = lambda x,y :  2.+np.sin(4.*np.pi*np.sqrt((x-0.6-0.25*0.)**2+(y-0.6)**2)) 

# ... test butterfly
#rho       = lambda x,y : 1.+7.*np.exp(-50.*abs((x)**2+(y-0.25*0.)**2-0.09)) 

#~~~~~~~~~~~~~~~
# Adapted mesh  
#~~~~~~~~~~~~~~~~~~~~
#---------------------------------------------------------
fig =plt.figure() 
for i in range(nbpts):
   phidx = ux[:,i]
   phidy = uy[:,i]

   plt.plot(phidx, phidy, '-b', linewidth = 0.25)
for i in range(nbpts):
   phidx = ux[i,:]
   phidy = uy[i,:]

   plt.plot(phidx, phidy, '-b', linewidth = 0.25)
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
#axes[0].axis('off')
plt.margins(0,0)
fig.tight_layout()
plt.savefig('figs/adaptive_meshes.png')
plt.show(block=False)
plt.close()

Z = rho(F1, F2)
fig, axes =plt.subplots() 
im2 = plt.contourf( F1, F2, Z, np.linspace(np.min(Z),np.max(Z),100), cmap= 'jet')
#divider = make_axes_locatable(axes) 
#cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
#plt.colorbar(im2, cax=cax) 
fig.tight_layout()
plt.savefig('figs/density_function.png')
plt.show(block=False)
plt.close()
