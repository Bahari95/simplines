from   .               import results_f90_core as core

#==============================================================
from numpy import zeros, linspace, meshgrid
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def pyccel_sol_field_2d( Npoints, uh, knots, degree, meshes = None, bound_val = None):
    '''
    Using computed control points uh we compute solution
    in new discretisation by Npoints
    The solution can be determined within the provided mesh
    The solution can be calculated within a particular domain : bound_val    
    '''
    pu, pv = degree
    Tu, Tv = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    
    if meshes is None:

	    if Npoints is None:

	       nx = nu-pu+1
	       ny = nv-pv+1
	    
	       xs = Tu[pu:-pu] 
	       ys = Tv[pv:-pv] 
	      
	    else :
	       '''
	       x0_v  : min val in x direction
	       x1_v  : max val in x direction
	       y0_v  : min val in y direction
	       y1_v  : max val in y direction
	       '''
	       nx, ny  = Npoints
	       if bound_val is not None:

		       x0_v = bound_val[0]
		       x1_v = bound_val[1] 
		       y0_v = bound_val[2] 
		       y1_v = bound_val[3]

	       else :
		       x0_v = Tu[pu]
		       x1_v = Tu[-pu-1]
		       y0_v = Tv[pv]
		       y1_v = Tv[-pv-1]
	       # ...
	       xs                     = linspace(x0_v, x1_v, nx)
	       ys                     = linspace(y0_v, y1_v, ny)
	    # ...
	    Q    = zeros((nx, ny, 3)) 
	    core.sol_field_2D(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, Q)
	    # ...
	    X, Y = meshgrid(xs, ys)
	    return Q[:,:,0], Q[:,:,1], Q[:,:,2], X.T, Y.T
    else :
       # ...
       nx, ny   = meshes[0].shape
       Q        = zeros((nx, ny, 5))
       Q[:,:,3] = meshes[0][:,:]
       Q[:,:,4] = meshes[1][:,:] 
       core.sol_field_2D_meshes(nx, ny, uh, Tu, Tv, pu, pv, Q)       
       return Q[:,:,0], Q[:,:,1], Q[:,:,2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def pyccel_sol_field_3d(Npoints,  uh , knots, degree, meshes = None):
    '''
    Using computed control points U we compute solution
    in new discretisation by Npoints    
    '''

    pu, pv, pz = degree
    Tu, Tv, Tz = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    nz = len(Tz) - pz - 1    
    if meshes is None:
       if Npoints is None:

            nx = nu-pu+1
            ny = nv-pv+1
            nz = nz-pz+1
    
            xs = Tu[pu:-pu] #linspace(Tu[pu], Tu[-pu-1], nx)
    
            ys = Tv[pv:-pv] #linspace(Tv[pv], Tv[-pv-1], ny)
      
            zs = Tz[pz:-pz] #linspace(Tv[pv], Tv[-pv-1], ny)
      
       else :
            nx, ny, nz = Npoints

            xs = linspace(Tu[pu], Tu[-pu-1], nx)
          
            ys = linspace(Tv[pv], Tv[-pv-1], ny)
       
            zs = linspace(Tz[pz], Tz[-pz-1], nz)
       Q    = zeros((nx, ny, nz, 7)) 
       core.sol_field_3D(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tz, pu, pv, pz, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3], Q[:,:,:,4], Q[:,:,:,5], Q[:,:,:,6]
    else :
       nx, ny, nz = Npoints
       # ...
       Q          = zeros((nx, ny, nz, 7))
       Q[:,:,:,4] = meshes[0][:,:,:]
       Q[:,:,:,5] = meshes[1][:,:,:]
       Q[:,:,:,6] = meshes[2][:,:,:]
       core.sol_field_3D_mesh(nx, ny, nz, uh, Tu, Tv, Tz, pu, pv, pz, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes L2 projection of 1D function
def least_square_Bspline(degree, knots, f, V_mae = None, x_mae = None, vec_in = None, y = None, m = None):
    from numpy     import zeros, linspace
    from .bsplines import find_span
    from .bsplines import basis_funs
    from scipy.sparse import csc_matrix, linalg as sla
    
    n       = len(knots) - degree - 1
    Tu      = knots[degree:degree+n]
    
    if m is None : 
        # ... in the case where f is a function

        m       = n + degree + 100 
        u_k     = linspace(knots[0], knots[degree+n], m)
        #...x_mae is not None implies that the Boudary conditions are fulfilled after applying the mapping to the boundary points
        if x_mae is not None :
           if y is None:
                u_kmae = pyccel_sol_field_2d((m,m),  x_mae , V_mae.knots, V_mae.degree)[0][vec_in,:]
           else :
                u_kmae = pyccel_sol_field_2d((m,m),  x_mae , V_mae.knots, V_mae.degree)[0][:,vec_in]

        # ...
        Pc      = zeros(n)
        Q       = zeros(m)
        if x_mae is not None :
           for i in range(0,m):
               Q[i] = f(u_kmae[i])
        else :
           for i in range(0,m):
               Q[i] = f(u_k[i])
    else : 
        # .. in the case of f is a vector
        # ...
        u_k     = linspace(knots[0], knots[degree+n], m)
        Pc      = zeros(n)
        Q       = zeros(m)
        if x_mae is not None:
           for i in range(0,m):
               Q[i] = f(x_mae[i])
        else :        
               Q[:] = f[:]
    
    Pc[0]   = Q[0]
    Pc[n-1] = Q[m-1]  
    #... Assembles matrix N of non vanishing basis functions in each u_k value
    N       = zeros((m-2,n-2))
    for k in range(1, m-1):
       span                           = find_span( knots, degree, u_k[k] )
       b                              = basis_funs( knots, degree, u_k[k], span )
       if span-degree ==0 :
          N[k-1,span-degree:span]     = b[1:]
       elif span+1 == n :
          N[k-1,span-degree-1:span-1] = b[:-1]
       else :
          N[k-1,span-degree-1:span]   = b

    #... Right hand side of least square Approximation
    R       = zeros(m-2)
    for k in range(1,m-1) : 
       span            = find_span( knots, degree, u_k[k] )
       b               = basis_funs( knots, degree, u_k[k], span )
       R[k-1] = Q[k]
       if span - degree == 0 :
          R[k-1]      -= b[0]*Q[0]
       if span + 1 == n :
          R[k-1]      -= b[degree]*Q[m-1]
    R      = N.T.dot(R)
    M      = (N.T).dot(N)
    #print(N,'\n M = ',M)
    lu       = sla.splu(csc_matrix(M))
    Pc[1:-1] = lu.solve(R)    
    return Pc

import matplotlib.pyplot            as     plt
from   mpl_toolkits.axes_grid1      import make_axes_locatable
import numpy                        as     np
colors = ['b', 'k', 'r', 'g', 'm', 'c', 'y', 'orange']
markers = ['v', 'o', 's', 'D', '^', '<', '>', '*']  # Different markers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_SolutionMultipatch(nbpts, xuh, V, xmp, ymp, savefig = None, plot = True): 
   ''''
   Plot the solution of the problem in the whole domain
   '''
   #---Compute a solution
   numPaches = len(V)
   u = []
   F1 = []
   F2 = []
   for i in range(numPaches):
      u.append(pyccel_sol_field_2d((nbpts, nbpts), xuh[i], V[i].knots, V[i].degree)[0])
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max, 100)  # Uniform levels for both plots

   # --- Create Figure ---
   fig, axes = plt.subplots(figsize=(8, 6))

   # --- Contour Plot for First Subdomain ---
   im = []
   for i in range(numPaches):
      im.append(axes.contourf(F1[i], F2[i], u[i], levels, cmap='jet'))
      # --- Colorbar ---
      divider = make_axes_locatable(axes)
      cax = divider.append_axes("right", size="5%", pad=0.05, aspect=40)
      cbar = plt.colorbar(im[i], cax=cax)
      cbar.ax.tick_params(labelsize=15)
      cbar.ax.yaxis.label.set_fontweight('bold')
   # --- Formatting ---
   axes.set_title("Solution the in whole domain ", fontweight='bold')
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold')

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_MeshMultipatch(nbpts, V, xmp, ymp, savefig = None, plot = True): 
   ''''
   Plot the solution of the problem in the whole domain
   '''
   #---Compute a solution
   numPaches = len(V)
   F1 = []
   F2 = []
   for i in range(numPaches):
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])

   # --- Create Figure ---
   fig =plt.figure() 

   # ---
   for ii in range(numPaches):
      #---------------------------------------------------------
      for i in range(nbpts):
         phidx = F1[ii][:,i]
         phidy = F2[ii][:,i]

      plt.plot(phidx, phidy, color = colors[ii], linewidth = 0.25)
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

      plt.plot(phidx, phidy, color = colors[ii], linewidth = 0.25)
      plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      phidx = F1[ii][:,0]
      phidy = F2[ii][:,0]
      plt.plot(phidx, phidy, 'm', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
      # ...
      phidx = F1[ii][:,nbpts-1]
      phidy = F2[ii][:,nbpts-1]
      plt.plot(phidx, phidy, 'b', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      #''
      phidx = F1[ii][0,:]
      phidy = F2[ii][0,:]
      plt.plot(phidx, phidy, 'r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
      # ...
      phidx = F1[ii][nbpts-1,:]
      phidy = F2[ii][nbpts-1,:]
      plt.plot(phidx, phidy, 'g', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0