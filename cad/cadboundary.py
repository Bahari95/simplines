"""
cadboundary.py

# Suitable parameterization from analytic CAD boundary information

@author : M. BAHARI
"""

from simplines          import compile_kernel
from simplines          import SplineSpace
from simplines          import TensorSpace
from simplines          import StencilVector
from simplines          import pyccel_sol_field_2d
from simplines          import least_square_Bspline
from simplines          import save_geometry_to_xml

from gallery_section_04 import assemble_stiffnessmatrix1D
from gallery_section_04 import assemble_massmatrix1D
from gallery_section_04 import assemble_vector_ex01
from gallery_section_04 import assemble_vector_ex02

assemble_stiffness  = compile_kernel( assemble_stiffnessmatrix1D, arity=2)
assemble_mass       = compile_kernel( assemble_massmatrix1D, arity=2)
assemble_rhs01      = compile_kernel(assemble_vector_ex01, arity=1)
assemble_rhs10      = compile_kernel(assemble_vector_ex02, arity=1)

#from numpy import asarray
import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
from   scipy.sparse                 import csr_matrix
from   numpy                        import zeros, linalg, asarray, cos, sin, pi, exp, sqrt
from   simplines                    import Poisson
from   scipy.sparse                 import kron
import numpy                        as     np
# ---
import time
import argparse
#==============================================================================
#  for figures 
import os
# Create the folder
os.makedirs("figs", exist_ok=True)  # 'exist_ok=True' prevents errors if the folder already exists


#.......Picard BFO ALGORITHM
def picard(V1, V2, V, u01, u10, u_01= None, u_10= None, x0 = None, y0 = None, niter = None):

    if niter is None:
       niter = 5
       i     = 0
       l2_residual = 10.
    tol = 1e-10
    from numpy import zeros
     
    Mx = zeros(V.nbasis)
    My = zeros(V.nbasis)
    
    #... We delete the first and the last spline function
    #. as a technic for applying Dirichlet boundary condition

    #..Stiffness and Mass matrix in 1D in the first deriction
    K1 = assemble_stiffness(V1)
    K1 = K1.tosparse()
    K1 = K1.toarray()[1:-1,1:-1]
    K1 = csr_matrix(K1)

    M1 = assemble_mass(V1)
    M1 = M1.tosparse()
    M1 = M1.toarray()

    # Stiffness and Mass matrix in 1D in the second deriction
    K2 = assemble_stiffness(V2)
    K2 = K2.tosparse()
    K2 = K2.toarray()[1:-1,1:-1]
    K2 = csr_matrix(K2)

    M2 = assemble_mass(V2)
    M2 = M2.tosparse()
    M2 = M2.toarray()
    # ...
    # .. 2D mass matrix for residual
    M_res  = kron(M1, M2)
    
    # ...
    M1 = csr_matrix(M1[1:-1,1:-1])
    M2 = csr_matrix(M2[1:-1,1:-1])
        
    mats_1 = [M1, K1]
    mats_2 = [M2, K2]

    # ...
    poisson = Poisson(mats_1, mats_2)
    minZ0 = -1.
    for rho_c in [1e-1,1.,2.3, 5., 7.5, 10.]:
        # ---------------------
        # .... Harmonic mapping
        # ---------------------
        du = StencilVector(V.vector_space)
        x1 = zeros(V1.nbasis*V2.nbasis)
        u_01 = StencilVector(V.vector_space)   
        u_10 = StencilVector(V.vector_space)
        #....
        rhs = assemble_rhs01( V , fields=[u_01, u_10, u01], value = [rho_c])
        b   = rhs.toarray()
        
        b   = b.reshape(V.nbasis) 
        b   = b[1:-1, 1:-1]      
        b   = b.reshape((V1.nbasis-2)*(V1.nbasis-2))

        #...
        xkron  = poisson.solve(b)
        #...
        xkron  = xkron.reshape([V1.nbasis-2,V2.nbasis-2])
        x01    = zeros(V.nbasis)
        x01[1:-1, 1:-1] = xkron[:, :]
        # ...Dirichlet
        x01[:, :] += x0[:, :]
        
        #--Assembles a right hand side of Poisson equation
        rhs = assemble_rhs10( V , fields=[u_01, u_10, u10], value = [rho_c])
        b   = rhs.toarray()
        b   = b.reshape(V.nbasis) 
        b   = b[1:-1, 1:-1]      
        b   = b.reshape((V1.nbasis-2)*(V2.nbasis-2))
        
        #...
        xkron  = poisson.solve(b)
        #...
        xkron  = xkron.reshape([V1.nbasis-2,V2.nbasis-2])
        x10    = zeros(V.nbasis)
        x10[1:-1, 1:-1] = xkron
        # ...Dirichlet
        x10[:, :] += y0[:, :]
        
        #... update the unkowns
        u_01.from_array(V, x01) 
        u_10.from_array(V, x10)         
        #//        
        dx = x10.reshape(V1.nbasis*V2.nbasis)-x1
        x1 = x10.reshape(V1.nbasis*V2.nbasis)
        Ja, Jb = pyccel_sol_field_2d((200,200),  x01 , V.knots, V.degree)[1:-2]
        Jc, Jd = pyccel_sol_field_2d((200,200),  x10 , V.knots, V.degree)[1:-2]
        Z      = Ja*Jd-Jb*Jc
        if np.min(Z) > minZ0:
            # ...Updat optimal mappng
            print(0,'a mapping is updeted using rho ',  rho_c,'with minJ =',np.min(Z))
            minZ0   = np.min(Z)
            Mx[:,:] = x01[:,:]
            My[:,:] = x10[:,:]
        for i in range(niter):
             #--Assembles a right hand side of Poisson equation
             rhs = assemble_rhs01( V , fields=[u_01, u_10, u01], value = [rho_c])
             b   = rhs.toarray()
        
             b   = b.reshape(V.nbasis) 
             b   = b[1:-1, 1:-1]      
             b   = b.reshape((V1.nbasis-2)*(V1.nbasis-2))
     
             #...
             xkron  = poisson.solve(b)
             #...
             xkron  = xkron.reshape([V1.nbasis-2,V2.nbasis-2])
             x01    = zeros(V.nbasis)
             x01[1:-1, 1:-1] = xkron[:, :]
             # ...Dirichlet
             x01[:, :] += x0[:, :]
        
             #//
             #--Assembles a right hand side of Poisson equation
             rhs = assemble_rhs10( V , fields=[u_01, u_10, u10], value = [rho_c])
             b   = rhs.toarray()
             b   = b.reshape(V.nbasis) 
             b   = b[1:-1, 1:-1]      
             b   = b.reshape((V1.nbasis-2)*(V2.nbasis-2))
        
             #...
             xkron  = poisson.solve(b)
             #...
             xkron  = xkron.reshape([V1.nbasis-2,V2.nbasis-2])
             x10    = zeros(V.nbasis)
             x10[1:-1, 1:-1] = xkron
             # ...Dirichlet
             x10[:, :] += y0[:, :]
             
             #... update the unkowns
             u_01.from_array(V, x01) 
             u_10.from_array(V, x10)         
             #//        
             dx          = x10.reshape(V1.nbasis*V2.nbasis)-x1
             x1          = x10.reshape(V1.nbasis*V2.nbasis)
             l2_residual = sqrt(dx.dot(M_res.dot(dx)) )
        
             Ja, Jb = pyccel_sol_field_2d((200,200),  x01 , V.knots, V.degree)[1:-2]
             Jc, Jd = pyccel_sol_field_2d((200,200),  x10 , V.knots, V.degree)[1:-2]
             Z      = Ja*Jd-Jb*Jc
             if np.min(Z) > minZ0:
                 # ...Updat optimal mappng
                 print(i,'a mapping is updeted using rho ',  rho_c,'with minJ =',np.min(Z))
                 minZ0   = np.min(Z)
                 Mx[:,:] = x01[:,:]
                 My[:,:] = x10[:,:]
                 
             if l2_residual < tol :
                 break
    #... update the unkowns
    u_01.from_array(V, Mx) 
    u_10.from_array(V, My)  
    #print(b)
    return u_01, u_10, Mx, My, i, l2_residual

# Check if the user provided the expressions as arguments
print("Usage: python3 cadboundary.py --expr1 'x' --expr2 'y'")
#..general geomitry 00 rho = 1.
"""
# Example 1
'0.15*sin(2.*pi*(y+0.1))*(x)+x' ' 0.15*sin(2.*pi*(x+0.1))*(y)+y
# Example 2
'0.15*sin(2.*pi*(y+0.1))*(x+0.5)+x' '0.15*cos(2.*pi*(x+0.1))*(y+0.5)+y'
# Example 3
'0.1*sin(3.*pi*(y+0.1))*(x+0.5)+x' '0.1*sin(3.*pi*(x+0.1))*(y+0.5)+y'
# Example 4
'0.15*sin(1.65*pi*(y+0.1))*(x+0.3)+x+0.3' '0.15*sin(1.65*pi*(x+0.1))*(y+0.3)+y+0.3'
# Example 5
'0.15*sin(2.75*pi*(y+0.1))*(x)+x' '0.15*sin(2.75*pi*(x+0.1))*(y)+y'
# Example 6
'0.18*sin(2.*pi*(y+0.1))*(x)+x' '0.18*sin(2.*pi*(x+0.1))*(y)+y'
# Example 7
'0.1*sin(3.*(1.75)*pi*(y+0.1))*(x+0.5)+x' '0.1*cos(4.*pi*(x+0.1))*(y+0.5)+y'
# Example 8 Quart annulus
'(0.2+0.8*x)*sin(.5*pi*y)' '(0.2+0.8*x)*cos(.5*pi*y)' 
# Example 9 circle
'(2.*x-1.)*sqrt(1.-(2.*y-1.)**2/2.0)' '(2.*y-1.)*sqrt(1.-(2.*x-1.)**2/2.0)'
"""
parser = argparse.ArgumentParser(description="Control plot behavior and save control points.")
parser.add_argument("--plot", action="store_true", help="Enable plotting and saving control points")
parser.add_argument("--degree", type=int, default=2, help="Degree of the polynomial (default: 2)")
parser.add_argument("--nelements", type=int, default=16, help="Number of elements (default: 16)")
parser.add_argument("--name", type=str, default='Geometry', help="Name of geometry (default: Geometry)")
parser.add_argument("--expr1", type=str, default="x", help="First mathematical expression (default: 'x')")
parser.add_argument("--expr2", type=str, default="y", help="Second mathematical expression (default: 'y')")
#.. If the boundary is given by four curves insted of analytic tranfsormation
parser.add_argument("--Xx0", type=str, default=None, help="mathematical expression in x direction (default: 'None')")
parser.add_argument("--Xx1", type=str, default=None, help="mathematical expression in x direction (default: 'None')")
parser.add_argument("--Xy0", type=str, default=None, help="mathematical expression in x direction (default: 'None')")
parser.add_argument("--Xy1", type=str, default=None, help="mathematical expression in x direction (default: 'None')")
parser.add_argument("--Yx0", type=str, default=None, help="mathematical expression in y direction (default: 'None')")
parser.add_argument("--Yx1", type=str, default=None, help="mathematical expression in y direction (default: 'None')")
parser.add_argument("--Yy0", type=str, default=None, help="mathematical expression in y direction (default: 'None')")
parser.add_argument("--Yy1", type=str, default=None, help="mathematical expression in y direction (default: 'None')")
args = parser.parse_args()

# Get the mathematical expressions and integers from the command-line arguments
degree      = args.degree
nelements   = args.nelements

if args.Xx0 is None:
    # Create the two functions dynamically
    print("Random analytic mapping")
    sol_dx = lambda x,y : eval(args.expr1)
    sol_dy = lambda x,y : eval(args.expr2)
    #__
    fx0 = lambda y : sol_dx(0.,y) 
    fy0 = lambda x : sol_dx(x,0.)
    fy1 = lambda x : sol_dx(x,1.)
    fx1 = lambda y : sol_dx(1.,y) 
    #__
    gx0 = lambda y : sol_dy(0.,y) 
    gy0 = lambda x : sol_dy(x,0.)
    gy1 = lambda x : sol_dy(x,1.)
    gx1 = lambda y : sol_dy(1.,y)
else :
    #__
    fx0 = lambda y : eval(args.Xx0)
    fy0 = lambda x : eval(args.Xy0)
    fy1 = lambda x : eval(args.Xy1)
    fx1 = lambda y : eval(args.Xx1)
    #___
    gx0 = lambda y : eval(args.Yx0)
    gy0 = lambda x : eval(args.Yy0)
    gy1 = lambda x : eval(args.Yy1)
    gx1 = lambda y : eval(args.Yx1)  

#..... Initialisation and computing optimal mapping for 16*16
#----------------------
# create the spline space for each direction
V1 = SplineSpace(degree=degree, nelements= nelements, nderiv = 2)
V2 = SplineSpace(degree=degree, nelements= nelements, nderiv = 2)

# create the tensor space
Vh = TensorSpace(V1, V2)

u01   = StencilVector(Vh.vector_space)
u10   = StencilVector(Vh.vector_space)

xD = zeros(Vh.nbasis)
yD = zeros(Vh.nbasis)

xD[0, : ]                   = least_square_Bspline(V1.degree, V1.knots, fx0)
xD[nelements+degree-1, : ]  = least_square_Bspline(V1.degree, V1.knots, fx1)
xD[:,0]                     = least_square_Bspline(V1.degree, V1.knots, fy0)
xD[:, nelements+degree - 1] = least_square_Bspline(V1.degree, V1.knots, fy1)

yD[0, : ]                   = least_square_Bspline(V2.degree, V2.knots, gx0)
yD[nelements+degree-1, : ]  = least_square_Bspline(V2.degree, V2.knots, gx1)
yD[:,0]                     = least_square_Bspline(V2.degree, V2.knots, gy0)
yD[:, nelements+degree - 1] = least_square_Bspline(V2.degree, V2.knots, gy1)

#yD *=10.
u01.from_array(Vh, xD)
u10.from_array(Vh, yD)
#+++++++++++++++++++++++++++++++++++
print('++ NEW-FORMULATION--FOR---VOLUMETRIC-PARAMETERIZATION')
start = time.time()
u11_pH, u12_pH, x11uh, x12uh, i, l2_residual = picard(V1, V2, Vh, u01, u10, x0 = xD, y0 = yD)
#x12uh /= 10.
#u12_pH.from_array(Vh, x12uh)
cpu_time =  time.time()-start

# ... save a control points
#np.savetxt('Circlex_'+str(degree)+'_'+str(nelements)+'.txt', x11uh, fmt='%.20e')
#np.savetxt('Circley_'+str(degree)+'_'+str(nelements)+'.txt', x12uh, fmt='%.20e')
# ...
#________________________________________________________________________________________________________________________
print('->  degree = {}  nelement = {}  CPU-time = {}'.format(degree, nelements, cpu_time),'\n ')
print('number of iteration = ',i, 'l2_residual', l2_residual,'\n ')
#---Compute a solution
nbpts = 280
#---Solution in uniform mesh
#u  = pyccel_sol_field_2d((nbpts,nbpts),  x2uh , V11.knots, V11.degree)[1]
ux, a, b  = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh.knots, Vh.degree)[:-2]
uy, c, d  = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh.knots, Vh.degree)[:-2]

Z = a*d-b*c

#...We extend the boundary curve for more suitable parameterization with respect to y
if np.min(Z) < 0. :
    yD *=10.
    u01.from_array(Vh, xD)
    u10.from_array(Vh, yD)
    #+++++++++++++++++++++++++++++++++++
    print('++ NEW-FORMULATION--FOR---VOLUMETRIC-PARAMETERIZATION')
    start = time.time()
    u11_pH, u12_pH, x11uh, x12uh, i, l2_residual = picard(V1, V2, Vh, u01, u10, x0 = xD, y0 = yD)
    x12uh /= 10.
    yD   /=10.
    u12_pH.from_array(Vh, x12uh)
    cpu_time =  time.time()-start

    # ...
    #________________________________________________________________________________________________________________________
    print('->  degree = {}  nelement = {}  CPU-time = {}'.format(degree, nelements, cpu_time),'\n ')
    print('number of iteration = ',i, 'l2_residual', l2_residual,'\n ')
    #---Compute a solution
    nbpts = 280
    #---Solution in uniform mesh
    #u  = pyccel_sol_field_2d((nbpts,nbpts),  x2uh , V11.knots, V11.degree)[1]
    ux, a, b  = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh.knots, Vh.degree)[:-2]
    uy, c, d  = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh.knots, Vh.degree)[:-2]

    Z = a*d-b*c    
    #...We extend the boundary curve for more suitable parameterization with respect to x
    if np.min(Z) < 0. :
        xD *=10.
        u01.from_array(Vh, xD)
        u10.from_array(Vh, yD)
        #+++++++++++++++++++++++++++++++++++
        print('++ NEW-FORMULATION--FOR---VOLUMETRIC-PARAMETERIZATION')
        start = time.time()
        u11_pH, u12_pH, x11uh, x12uh, i, l2_residual = picard(V1, V2, Vh, u01, u10, x0 = xD, y0 = yD)
        x11uh /= 10.
        xD   /=10.
        u11_pH.from_array(Vh, x11uh)
        cpu_time =  time.time()-start

        # ...
        #________________________________________________________________________________________________________________________
        print('->  degree = {}  nelement = {}  CPU-time = {}'.format(degree, nelements, cpu_time),'\n ')
        print('number of iteration = ',i, 'l2_residual', l2_residual,'\n ')
        #---Compute a solution
        nbpts = 280
        #---Solution in uniform mesh
        #u  = pyccel_sol_field_2d((nbpts,nbpts),  x2uh , V11.knots, V11.degree)[1]
        ux, a, b  = pyccel_sol_field_2d((nbpts,nbpts),  x11uh , Vh.knots, Vh.degree)[:-2]
        uy, c, d  = pyccel_sol_field_2d((nbpts,nbpts),  x12uh , Vh.knots, Vh.degree)[:-2]

        Z = a*d-b*c   

print( ' min of Jacobian in the intire unit square =', np.min(Z) )
print( ' max of Jacobian in the intire unit square =', np.max(Z) )

# ... save data
#np.savetxt('figs/filex_'+str(degree)+'_'+str(nelements)+'.txt', x11uh, fmt='%.20e')
#np.savetxt('figs/filey_'+str(degree)+'_'+str(nelements)+'.txt', x12uh, fmt='%.20e')
Gmap  = np.zeros((V1.nbasis*V2.nbasis,2))
x11uh = x11uh.reshape(V1.nbasis*V2.nbasis)
x12uh = x12uh.reshape(V1.nbasis*V2.nbasis)
Gmap[:,0] = x11uh[:]
Gmap[:,1] = x12uh[:]
save_geometry_to_xml(Vh, Gmap, name = args.name)
#np.savetxt('figs/Geom_'+str(degree)+'_'+str(nelements)+'.txt', Gmap, fmt='%.20e')
if args.plot :
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
    plt.plot(u11_pH.toarray(), u12_pH.toarray(), 'ro', markersize=3.5)
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

    #axes[0].axis('off')
    plt.margins(0,0)
    fig.tight_layout()
    plt.savefig('figs/meshes.png')
    plt.show(block=False)
    plt.close()

    fig, axes =plt.subplots() 
    im2 = plt.contourf( ux, uy, Z, np.linspace(np.min(Z),np.max(Z),100), cmap= 'jet')
    divider = make_axes_locatable(axes) 
    cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
    plt.colorbar(im2, cax=cax) 
    fig.tight_layout()
    plt.savefig('figs/Jacobian.png')
    plt.show(block=False)
    plt.close()
else:
    print("Plotting is disabled. No files were saved, re-run with --plot")
