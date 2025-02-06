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
#from   kronecker.fast_diag          import Poisson
# ...   load a geometry from xml file 
from   simplines                    import getGeometryMap

from gallery_section_06             import assemble_stiffnessmatrix1D
from gallery_section_06             import assemble_massmatrix1D
from gallery_section_06             import assemble_matrix_ex01
from gallery_section_06             import assemble_matrix_ex02
#..
from gallery_section_06             import assemble_vector_ex01
from gallery_section_06             import assemble_Quality_ex01

assemble_stiffness1D = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass1D      = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_matrix_ex01 = compile_kernel(assemble_matrix_ex01, arity=1)
assemble_matrix_ex02 = compile_kernel(assemble_matrix_ex02, arity=1)
#..
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


#==============================================================================
#.......Poisson ALGORITHM
def picard_solve(V1, V2, V3, V4, V,  V00, V11, V01, V10, u11_mpH = None, u12_mpH = None, times = None, x_2 = None, tol = None):
       niter      = 30   #
       if tol is None :
          tol     = 1e-8  # 
       # .. computes basis and sopans in adapted quadrature
       Quad_adm   = quadratures_in_admesh(V)
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
       del x_D
       del y_D

       #___
       I1         = np.eye(V3.nbasis)
       I2         = np.eye(V4.nbasis)

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
       r1         = R1.T
       R1         = R1[1:-1,:].T
       R1         = csr_matrix(R1)
       r1         = csr_matrix(r1)
       #___
       R2         = assemble_matrix_ex02(V10)
       R2         = R2.toarray()
       R2         = R2.reshape(V10.nbasis)
       r2         = R2
       R2         = R2[:,1:-1]
       R2         = csr_matrix(R2)
       r2         = csr_matrix(r2)

       #...step 0.1
       mats_1     = [M1, M1]
       mats_2     = [D2, D2]

       # ...Fast Solver
       poisson_c1 = Poisson(mats_1, mats_2)
              
       #...step 0.2
       mats_1     = [D1, D1]
       mats_2     = [M2, M2]

       # ...Fast Solver
       poisson_c2 = Poisson(mats_1, mats_2)
       
       #...step 1
       M1         = sla.inv(M1) #... I don't know if I can avoid the inverse TODO
       A1         = M1.dot(R1.T)
       K1         = R1.dot( A1)
       K1         = csr_matrix(K1)
       #___
       M2         = sla.inv(M2)
       A2         = M2.dot( R2.T)
       K2         = R2.dot( A2)
       K2         = csr_matrix(K2)

       #...step 2
       mats_1     = [D1, K1]
       mats_2     = [D2, K2]

       # ...Fast Solver
       poisson    = Poisson(mats_1, mats_2)

       #...non homogenoeus Neumann boundary 
       b01        = -kron(r1, D2).dot(u_01.toarray())
       #__
       b10        = -kron(D1, r2).dot( u_10.toarray())
       b_0        = b01 + b10
       #...
       b11        = -kron(m1[1:-1,:], D2).dot(u_01.toarray())
       #___
       b12        = -kron(D1, m2[1:-1,:]).dot(u_10.toarray())
       
       #___Solve first system
       r_0        =  kron(A1.T, I2).dot(b11) + kron(I1, A2.T).dot(b12)

       #___
       x11_1      = kron(A1, I2)
       x12_1      = kron(I1, A2)
       #___
       C1         = poisson_c1.solve(2.*b11)
       C2         = poisson_c2.solve(2.*b12)

       # ... for Two or Multi grids
       if x_2 is None :    
          u11     = StencilVector(V01.vector_space)
          u12     = StencilVector(V10.vector_space)
          x11     = np.zeros(V01.nbasis) # dx/ appr.solution
          x12     = np.zeros(V10.nbasis) # dy/ appr.solution
          # ...
          u11.from_array(V01, x11)
          u12.from_array(V10, x12)
          # ...

          # .../
          x_2     = zeros(V3.nbasis*V4.nbasis)
       else           :       
          #print( 'one pice is true')
          u11          = StencilVector(V01.vector_space)
          u12          = StencilVector(V10.vector_space)
          x11          = np.zeros(V01.nbasis) # dx/ appr.solution
          x12          = np.zeros(V10.nbasis) # dy/ appr.solution
          # ...Assembles Neumann (Dirichlet) boundary conditions
          x11[-1,:]    = 1.
          x12[:,-1]    = 1.
          # ...
          x11[1:-1,:]  =  (C1 - x11_1.dot(x_2)).reshape([V1.nbasis-2,V3.nbasis])
          u11.from_array(V01, x11)
          #___
          x12[:,1:-1]  =  (C2 - x12_1.dot(x_2)).reshape([V4.nbasis,V2.nbasis-2])
          u12.from_array(V10, x12)      
       # ... for assembling residual
       M_res      = kron(D1, D2)
       #___       
       del poisson_c1
       del poisson_c2
       del b01
       del b10
       del R1
       del R2
       del r1
       del r2
       del D1
       del D2
       del M1
       del M2
       del A1
       del A2
       del b11
       del b12
       
       for i in range(niter):
           
           # ... computes spans and basis in adapted quadrature 
           spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11, u12)
           #---Assembles a right hand side of Poisson equation
           rhs          = StencilVector(V11.vector_space)
           rhs          = assemble_rhs(V, fields = [u11, u12, u11_mpH, u12_mpH], value = [spans_ad1, spans_ad2, basis_ad1, basis_ad2], out= rhs)
           b            = rhs.toarray()
           b            = b_0 + b.reshape(V4.nbasis*V3.nbasis)
           #___
           r            =  r_0 - b
           
           # ... Solve first system
           x2           = poisson.solve(r)
           x2           = x2 -sum(x2)/len(x2)
           #___
           x11[1:-1,:]  =  (C1 - x11_1.dot(x2)).reshape([V1.nbasis-2,V3.nbasis])
           u11.from_array(V01, x11)
           #___
           x12[:,1:-1]  =  (C2 - x12_1.dot(x2)).reshape([V4.nbasis,V2.nbasis-2])
           u12.from_array(V10, x12)

           #..Residual 
           dx           = x2[:]-x_2[:]
           x_2[:]       = x2[:]
           
           #... Compute residual for L2
           l2_residual   = sqrt(dx.dot(M_res.dot(dx)) )
           
           if i == 0 :    
             # ...Assembles Neumann boundary conditions
             x11[-1,:]  = 1.
             x12[:,-1]  = 1.
             # ...
             u11.from_array(V01, x11)
             u12.from_array(V10, x12)
           if l2_residual < tol:
              break
       return u11, u12, x11, x12, i, l2_residual, x_2

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
   V1mpH          = SplineSpace(degree=degree-1,   nelements= Hnelements, nderiv = 2, quad_degree = degree)
   V2mpH          = SplineSpace(degree=degree-1,   nelements= Hnelements, nderiv = 2, quad_degree = degree)	
   VHmp           = TensorSpace(V1mpH, V2mpH)
   # ... Assembling mapping
   mp             = getGeometryMap(geometry,0)
   xmp, ymp       = mp.coefs()

   xmp = xmp.reshape(VHmp.nbasis)
   ymp = ymp.reshape(VHmp.nbasis)

   u11_mpH        = StencilVector(VHmp.vector_space)
   u12_mpH        = StencilVector(VHmp.vector_space)
   u11_mpH.from_array(VHmp, xmp)
   u12_mpH.from_array(VHmp, ymp)

   # ... G-space
   VH             = TensorSpace(V1H, V2H, V3H, V4H, V1mpH, V2mpH)

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
      V1mph       = SplineSpace(degree=degree-1,   nelements= nelements, nderiv = 2, quad_degree = degree)
      V2mph       = SplineSpace(degree=degree-1,   nelements= nelements, nderiv = 2, quad_degree = degree)	
      Vhmp        = TensorSpace(V1mph, V2mph)
         
      Vhmg        = TensorSpace(V1mg, V2mg, V3mg, V4mg, V1mph, V2mph)

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
   V1mph           = SplineSpace(degree=degree-1,   nelements= nelements, nderiv = 2, quad_degree = degree)
   V2mph           = SplineSpace(degree=degree-1,   nelements= nelements, nderiv = 2, quad_degree = degree)	
   Vhmp            = TensorSpace(V1mph, V2mph)
         
   Vh              = TensorSpace(V1, V2, V3, V4, V1mph, V2mph)
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

   # ... in fine grid
   start            = time.time()
   u11_pH, u12_pH, x11uh, x12uh, iter_N, l2_residualh = picard_solve(V1, V2, V3, V4, Vh, Vh00, Vh11, Vh01, Vh10, u11_mpH = u11_mph, u12_mpH = u12_mph, times = times, x_2 = x2H)[:-1]
   MG_time         += time.time()- start
   # ...
   # .. computes basis and sopans in adapted quadrature
   Quad_adm         = quadratures_in_admesh(Vh)
   spans_ad1, spans_ad2, basis_ad1, basis_ad2 = Quad_adm.ad_quadratures(u11_pH, u12_pH)
   Quality          = StencilVector(Vh11.vector_space)
   Quality          = assemble_Quality(Vh, fields=[u11_pH, u12_pH, u11_mph, u12_mph], value = [times, spans_ad1, spans_ad2, basis_ad1, basis_ad2],  out = Quality)
   norm             = Quality.toarray()
   l2_Quality       = norm[0]
   l2_displacement  = norm[1]
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
	#geometry = '../fields/IP.xml'

	# ... Butterfly
	geometry = '../fields/butterfly.xml'

	# ... nelement = 2**nb_ne
	nb_ne           = 6
	
	nelements, l2_Quality, MG_time, l2_displacement, x11uh , Vh01, x12uh , Vh10, xmp, ymp, Vhmp = Monge_ampere_equation(nb_ne, geometry= geometry, check = True)

	#---Compute a solution
	nbpts              = 100
	
	#---Solution in uniform mesh
	sx, uxx, uxy, X, Y = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh01.knots, Vh01.degree)
	sy, uyx, uyy       = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh10.knots, Vh10.degree)[0:3]

	#---Compute a mapping
	F1 = pyccel_sol_field_2d((nbpts,nbpts),  xmp , Vhmp.knots, Vhmp.degree)[0]
	F2 = pyccel_sol_field_2d((nbpts,nbpts),  ymp , Vhmp.knots, Vhmp.degree)[0]
	# ... in adaped mesh
	ux = pyccel_sol_field_2d( None,  xmp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
	uy = pyccel_sol_field_2d( None,  ymp , Vhmp.knots, Vhmp.degree, meshes = (sx, sy))[0]
	# ... Jacobian function of Optimal mapping
	det = uxx*uyy-uxy**2

	det_min          = np.min( det[1:-1,1:-1])
	det_max          = np.max( det[1:-1,1:-1])
	#... tabulate 

	print("degree = ", Vh01.degree[0])
	table            = [ [Vh01.degree[0], nelements, l2_Quality, MG_time, l2_displacement, det_min, det_max] ]
	headers          = ["degree", "$#$cells"," Err","CPU-time (s)", "Qual" ,"$min-det(PsiPsi)$", "$max_det(PsiPsi)$"]
	print(tabulate(table, headers, tablefmt="github"),'\n')

if False :
	V1mpH          = SplineSpace(degree=2,   nelements= 16)
	V2mpH          = SplineSpace(degree=2,   nelements= 16)	
	VHmp           = TensorSpace(V1mpH, V2mpH)
	xm             = np.loadtxt('fields/Butterflyx_'+str(2)+'_16.txt')*0.6+0.5
	ym             = np.loadtxt('fields/Butterflyy_'+str(2)+'_16.txt')*0.6+0.5
	nxm  = pyccel_sol_field_2d( None,  x11uh , Vh01.knots, Vh01.degree, meshes = (xm, ym))[0]
	nym  = pyccel_sol_field_2d( None,  x12uh , Vh10.knots, Vh10.degree, meshes = (xm, ym))[0]	
	X2 = pyccel_sol_field_2d((nbpts,nbpts),  nxm , VHmp.knots, VHmp.degree)[0]
	Y2 = pyccel_sol_field_2d((nbpts,nbpts),  nym , VHmp.knots, VHmp.degree)[0]	
	X1 = pyccel_sol_field_2d((nbpts,nbpts),  xm , VHmp.knots, VHmp.degree)[0]
	Y1 = pyccel_sol_field_2d((nbpts,nbpts),  ym , VHmp.knots, VHmp.degree)[0]
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
	
#~~~~~~~~~~~~~~~~~~~~~~~
for i in range(nbpts):
  for j in range(nbpts):
     if det[i,j] < 0.:
         print('Npoints =',nbpts,'min_Jac-F in the entire domain = ', det[i,j] ,'index =', i, j)

print('..../!\...: min~max value of the Jacobian function =', np.min(det),'~', np.max(det) )

#         -++++++++++++++++++++++++++++++++++++++++++++++++++++ End of sharing part of any geometry-----------------------------------------------------------

#.. Analytic Density function 
rho = lambda x,y : 1.+ 9./(1.+(10.*sqrt((x-0.5-0.25*0.)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.5-0.25*0.) -10.*((x-0.5-0.25*0.)**2+(y-0.5)**2)))**2)

#rho = lambda x,y :1+5*np.exp(-100*np.abs((x-0.45)**2+(y-0.4)**2-0.09))+5.*np.exp(-100.*np.abs(x**2+y**2-0.2))+5.*np.exp(-100*np.abs((x+0.45)**2 +(y-0.4)**2-0.1))+7.*np.exp(-100.*np.abs(x**2+(y+1.25)**2-0.4))
#.. Test 1  circle
#def rho(x,y):
#   return 1. + 5./np.cosh( 5.*((x-np.sqrt(3)/2)**2+(y-0.5)**2 - (np.pi/2)**2) )**2 + 5./np.cosh( 5.*((x+np.sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2


# ... test butterfly
#rho       = lambda x,y : 2.+np.sin(10.*np.pi*np.sqrt((x-0.6)**2+(y-0.6)**2)) 

fig =plt.figure() 
for i in range(nbpts):
   phidx = X[:,i]
   phidy = Y[:,i]

   plt.plot(phidx, phidy, '-k', linewidth = .3)
for i in range(nbpts):
   phidx = X[i,:]
   phidy = Y[i,:]

   plt.plot(phidx, phidy, '-k', linewidth = .3)
#plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
phidx = X[:,0]
phidy = Y[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
# ...
phidx = X[:,nbpts-1]
phidy = Y[:,nbpts-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
#''
phidx = X[0,:]
phidy = Y[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
# ...
phidx = X[nbpts-1,:]
phidy = Y[nbpts-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

#plt.xlim([-0.075,0.1])
#plt.ylim([-0.25,-0.1])
#axes[0].axis('off')
plt.margins(0,0)
fig.tight_layout()
plt.savefig('figs/Un_meshes_US.png')
plt.show(block=False)
plt.close()

fig =plt.figure() 
for i in range(nbpts):
   phidx = F1[:,i]
   phidy = F2[:,i]

   plt.plot(phidx, phidy, '-k', linewidth = .3)
for i in range(nbpts):
   phidx = F1[i,:]
   phidy = F2[i,:]

   plt.plot(phidx, phidy, '-k', linewidth = .3)
#plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
phidx = F1[:,0]
phidy = F2[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
# ...
phidx = F1[:,nbpts-1]
phidy = F2[:,nbpts-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
#''
phidx = F1[0,:]
phidy = F2[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
# ...
phidx = F1[nbpts-1,:]
phidy = F2[nbpts-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

#plt.xlim([-0.075,0.1])
#plt.ylim([-0.25,-0.1])
plt.axis('off')
plt.margins(0,0)
fig.tight_layout()
plt.savefig('figs/Un_meshes.png')
plt.show(block=False)
plt.close()

#~~~~~~~~~~~~~~~
# Adapted mesh  
#~~~~~~~~~~~~~~~~~~~~
fig =plt.figure() 
for i in range(nbpts):
   phidx = sx[:,i]
   phidy = sy[:,i]

   plt.plot(phidx, phidy, '-k', linewidth = .3)
for i in range(nbpts):
   phidx = sx[i,:]
   phidy = sy[i,:]

   plt.plot(phidx, phidy, '-k', linewidth = .3)
#plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
#~~~~~~~~~~~~~~~~~~~~
#.. Plot the surface
phidx = sx[:,0]
phidy = sy[:,0]
plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
# ...
phidx = sx[:,nbpts-1]
phidy = sy[:,nbpts-1]
plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
#''
phidx = sx[0,:]
phidy = sy[0,:]
plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
# ...
phidx = sx[nbpts-1,:]
phidy = sy[nbpts-1,:]
plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

#plt.xlim([-0.075,0.1])
#plt.ylim([-0.25,-0.1])
#axes[0].axis('off')
plt.margins(0,0)
fig.tight_layout()
plt.savefig('figs/adaptive_meshes_US.png')
plt.show(block=False)
plt.close()
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

fig =plt.figure() 
#axes[0].set_title( 'Physical domain ' )
for i in range(nbpts):
    phidx = ux[:,i]
    phidy = uy[:,i]

    plt.plot(phidx, phidy, '-b', linewidth = 0.54)
for i in range(nbpts):
    phidx = ux[i,:]
    phidy = uy[i,:]

    plt.plot(phidx, phidy, '-b', linewidth = 0.54)
#plt.axis('off')
#plt.margins(0,0)
i     = 0
phidx = F1[:,i]
phidy = F2[:,i]
plt.plot(phidx, phidy, '-r', linewidth = 2.)
i     = nbpts-1
phidx = F1[:,i]
phidy = F2[:,i]
plt.plot(phidx, phidy, '-r', linewidth = 2.)
#''
i     = 0
phidx = F1[i,:]
phidy = F2[i,:]
plt.plot(phidx, phidy, '-r', linewidth = 2.)
i     = nbpts-1
phidx = F1[i,:]
phidy = F2[i,:]
plt.plot(phidx, phidy, '-r', linewidth = 2.)
plt.xlim(0., 0.5)
plt.ylim(0.5, 1.)
plt.savefig('figs/close_up_adaptive_meshes.png')
plt.show(block=False)
plt.close()
