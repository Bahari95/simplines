"""
results_f90.py: A fast post-processing script for visualizing the solution and its derivatives.

Author: M. Mustapha Bahari
"""

from   .               import results_f90_core as core

#==============================================================
from numpy import zeros, linspace, meshgrid
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def pyccel_sol_field_1d(knots, uh, Npoints = None, meshes = None, bound_val = None):
   """
   Computes the solution and its gradient in one dimension.
   """
   Tu = knots
   nu = uh.shape[0]
   pu = len(Tu) - nu -1
   
   if meshes is None:

      if Npoints is None:
         nx     = nu-pu+1
         meshes = Tu[pu:-pu] 
      else :
         '''
         x0_v  : min val in x direction
         x1_v  : max val in x direction
         '''
         nx   = Npoints
         if bound_val is not None:
            x0_v = bound_val[0]
            x1_v = bound_val[1] 

         else :
            x0_v = Tu[pu]
            x1_v = Tu[-pu-1]
         # ...
         meshes  = linspace(x0_v, x1_v, nx)
   # ...
   nx       = meshes.shape[0]
   Q        = zeros((nx, 3))
   Q[:,2]   = meshes[:]
   core.sol_field_1D_meshes(nx, uh, Tu, pu, Q)
   return Q[:,0], Q[:,1]

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
def least_square_Bspline(degree, knots, f):
    """
    Computes the least squares projection of a 1D function or vector onto a B-spline basis.
    """
    from numpy     import zeros, linspace
    from .bsplines import find_span
    from .bsplines import basis_funs
    from scipy.sparse import csc_matrix, linalg as sla
    
    n       = len(knots) - degree - 1

    if callable(f):
        # ... in the case where f is a function
        m       = n + degree + 100
        u_k     = linspace(knots[0], knots[degree+n], m)
        # ...
        Pc      = zeros(n)
        Q       = zeros(m)
        for i in range(0,m):
            Q[i] = f(u_k[i])
    else:
        # .. in the case of f is a vector
        m       = len(f)
        u_k     = linspace(knots[0], knots[degree+n], m)
        Pc      = zeros(n)
        Q       = zeros(m)
        Q[:]    = f[:]
    
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
import pyvista                      as     pv
import os
colors = ['b', 'k', 'r', 'g', 'm', 'c', 'y', 'orange']
markers = ['v', 'o', 's', 'D', '^', '<', '>', '*']  # Different markers
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_SolutionMultipatch(nbpts, xuh, V, xmp, ymp, savefig = None, plot = True, Jacfield = None): 
   """
   Plot the solution of the problem in the whole multi-patch domain
   """
   #---Compute a solution
   numPaches = len(V)
   u   = []
   F1  = []
   F2  = []
   JF = []
   for i in range(numPaches):
      u.append(pyccel_sol_field_2d((nbpts, nbpts), xuh[i], V[i].knots, V[i].degree)[0])
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      #...Compute a Jacobian
      F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[1:3]
      F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[1:3]
      JF.append(F1x*F2y - F1y*F2x)
   if Jacfield is not None:
       u = JF

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots

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
def plot_JacobianMultipatch(nbpts, V, xmp, ymp, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   u   = []
   F1  = []
   F2  = []
   for i in range(numPaches):
      #---Compute a solution
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      #...Compute a Jacobian
      F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[1:3]
      F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[1:3]
      u.append(F1x*F2y - F1y*F2x)

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(u[0]), np.min(u[1]))
   u_max  = max(np.max(u[0]), np.max(u[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(u[i]))
      u_max  = max(u_max, np.max(u[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots

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
   #axes.set_title("Jacobian the in whole domain ", fontweight='bold')
   for label in axes.get_xticklabels() + axes.get_yticklabels():
      label.set_fontweight('bold')

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_MeshMultipatch(nbpts, V, xmp, ymp, cp = True, savefig = None, plot = True): 
   """
   Plot the solution of the problem in the whole domain
   """
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

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'k')
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

         plt.plot(phidx, phidy, linewidth = 0.5, color = 'k')
      if cp:
         plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      if ii == 1:
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '-g', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      else :
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '-g', linewidth=2., label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=2. ,label = '$Im([0,1]^2_{y=1})$')
      #''
      phidx = F1[ii][0,:]
      phidy = F2[ii][0,:]
      plt.plot(phidx, phidy, '-r',  linewidth=2., label = '$Im([0,1]^2_{x=0})$')
      # ...
      phidx = F1[ii][nbpts-1,:]
      phidy = F2[ii][nbpts-1,:]
      plt.plot(phidx, phidy, '-r', linewidth= 2., label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_FunctMultipatch(nbpts, V, xmp, ymp, Func, cp = True, savefig = None, plot = True): 
   """
   Plot the function in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1     = []
   F2     = []
   values = []
   for i in range(numPaches):
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0])
      F2.append(pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0])
      values.append(Func(F1[i], F2[i]))

   # --- Compute Global Color Levels ---
   u_min  = min(np.min(values[0]), np.min(values[1]))
   u_max  = max(np.max(values[0]), np.max(values[1]))
   for i in range(2, numPaches):
      u_min  = min(u_min, np.min(values[i]))
      u_max  = max(u_max, np.max(values[i]))
   levels = np.linspace(u_min, u_max+1e-10, 100)  # Uniform levels for both plots
   # --- Create Figure ---
   # ... Analytic Density function
   fig, axes =plt.subplots() 
   for i in range(numPaches):
      im2 = plt.contourf( F1[i], F2[i], values[i], levels, cmap= 'plasma')
   #divider = make_axes_locatable(axes) 
   #cax   = divider.append_axes("right", size="5%", pad=0.05, aspect = 40) 
   #plt.colorbar(im2, cax=cax) 
   fig.tight_layout()

   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def plot_AdMeshMultipatch(nbpts, V, xmp, ymp, xad, yad, cp = True, savefig = None, plot = True, patchesInterface = False): 
   """
   Plot the solution of the problem in the whole domain
   """
   #---Compute a solution
   numPaches = len(V)
   F1 = []
   F2 = []
   for i in range(numPaches):
      sx = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
      sy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
      #---Compute a mesh
      F1.append(pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0])
      F2.append(pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0])

   # --- Create Figure ---
   fig =plt.figure() 

   # ---
   for ii in range(numPaches):
      #---------------------------------------------------------
      for i in range(nbpts):
         phidx = F1[ii][:,i]
         phidy = F2[ii][:,i]

         plt.plot(phidx, phidy, linewidth = 0.3, color = 'k')
      for i in range(nbpts):
         phidx = F1[ii][i,:]
         phidy = F2[ii][i,:]

         plt.plot(phidx, phidy, linewidth = 0.3, color = 'k')
      if cp:
         plt.plot(xmp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), ymp[ii].reshape(V[ii].nbasis[0]*V[ii].nbasis[1]), 'ro', markersize=3.5)
      #~~~~~~~~~~~~~~~~~~~~
      #.. Plot the surface
      if patchesInterface:
         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=0.25, label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=0.25 ,label = '$Im([0,1]^2_{y=1})$')

         phidx = F1[ii][:,0]
         phidy = F2[ii][:,0]
         plt.plot(phidx, phidy, '--k', linewidth=0.25, label = '$Im([0,1]^2_{y=0})$')
         # ...
         phidx = F1[ii][:,nbpts-1]
         phidy = F2[ii][:,nbpts-1]
         plt.plot(phidx, phidy, '--k', linewidth=0.25,label = '$Im([0,1]^2_{y=1})$')
         #''
         phidx = F1[ii][0,:]
         phidy = F2[ii][0,:]
         plt.plot(phidx, phidy, '--k',  linewidth=0.25, label = '$Im([0,1]^2_{x=0})$')
         # ...
         phidx = F1[ii][nbpts-1,:]
         phidy = F2[ii][nbpts-1,:]
         plt.plot(phidx, phidy, '--k', linewidth= 0.25, label = '$Im([0,1]^2_{x=1}$)')

   #axes[0].axis('off')
   plt.margins(0,0)

   fig.tight_layout()
   if savefig is not None:
      plt.savefig(savefig)
   plt.show(block=plot)
   print('Plotting done :  Solution in the whole domain (type savefig = \'location/somthing.png\' to save the figure)')
   return 0

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# ... Post processing using Paraview 
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_AdMeshMultipatch(nbpts, V, xmp, ymp, xad, yad, zad = None, zmp = None, solution = None, Func = None, output_path = "figs/admultipatch_multiblock.vtm", plot = True): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   V : list
       List of patch objects containing spline spaces.
   xmp, ymp : list
       Lists of control points for the initial mapping in x and y directions.
   xad, yad : list
       Lists of control points for the adaptive mesh in x and y directions.
   zad : list, optional
       List of control points for the adaptive mesh in z direction (for 3D).
   zmp : list, optional
       List of control points for the initial mapping in z direction (for 3D).
   solution : list, optional
       List of solution control points for each patch and its name.
       solutions = [
         {"name": "displacement", "data": xuh = list},   # e.g., displacement field control points
         {"name": "velocity", "data": yuh = list},   # e.g., velocity field control points
         # Add more solution fields as needed
      ]
   Func : callable, optional
       Analytic function to evaluate on the mesh (signature depends on dimension).
   output_path : str, optional
       Path to save the output VTM file (default: "figs/admultipatch_multiblock.vtm").
   plot : bool, optional
       If True, enables plotting (not used in this function).

   Returns
   -------
   None
       The function saves the multi-block dataset to the specified output path.
   """
   numPaches = len(V)
   os.makedirs("figs", exist_ok=True)
   multiblock = pv.MultiBlock()

   #...
   #F3 = [] 
   if zmp is None:
      if solution is None:
         if Func is None:
            for i in range(numPaches):
               #... computes adaptive mesh
               sx, sxx, sxy = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               #...Compute a Jacobian
               #Jf = (F1x*F2y - F1y*F2x)
               Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #... computes adaptive mesh
               sx, sxx, sxy = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               #...Compute a Jacobian
               #Jf = (F1x*F2y - F1y*F2x)
               Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
               # ... image bu analytic function
               fnc  = Func(x, y)
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid

      else:
         if Func is None:
            for i in range(numPaches):
               #... computes adaptive mesh
               sx, sxx, sxy = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               #...Compute a Jacobian
               Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

               # .... 
               for sol in solution:
                  U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #... computes adaptive mesh
               sx, sxx, sxy = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               #...Compute a Jacobian
               #Jf = (F1x*F2y - F1y*F2x)
               Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
               # ... image bu analytic function
               fnc  = Func(x, y)
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   else:
      if zad is None :
         if solution is None:
            if Func is None:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx, sxx, sxy = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0:3]
                  sy, syx, syy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0:3]
                  #---Compute a image by initial mapping
                  x, F1x, F1y = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
                  y, F2x, F2y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
                  z           = pyccel_sol_field_2d((None, None), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  #...Compute a Jacobian in  i direction
                  #Jf = (F1x*F2y - F1y*F2x)
                  Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
                  # .... 
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, 1]

                  # Flatten the solution and attach as a scalar field
                  grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
            else:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
                  sy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  z = pyccel_sol_field_2d((None, None), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  # .... 
                  # ... image bu analytic function
                  fnc  = Func(x, y, z)
                  #...
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, 1]

                  # Flatten the solution and attach as a scalar field
                  grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid

         else:
            if Func is None:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
                  sy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  z = pyccel_sol_field_2d((None, None), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  #...
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, 1]

                  # Flatten the solution and attach as a scalar field
                  for sol in solution:
                     # .... 
                     U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
            else:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx = pyccel_sol_field_2d((nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
                  sy = pyccel_sol_field_2d((nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x = pyccel_sol_field_2d((None, None), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  y = pyccel_sol_field_2d((None, None), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  z = pyccel_sol_field_2d((None, None), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  # .... 
                  # ... image bu analytic function
                  fnc  = Func(x, y, z)
                  #...
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, 1]

                  # Flatten the solution and attach as a scalar field
                  grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
                  for sol in solution:
                     # .... 
                     U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
      else: #... zad
         if solution is None:
            if Func is None:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx, uxx, uxy, uxz = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0:4]
                  sy, uyx, uyy, uyz = pyccel_sol_field_3d((nbpts, nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0:4]
                  sz, uzx, uzy, uzz = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zad[i], V[i].knots, V[i].degree)[0:4]
                  #---Compute a image by initial mapping
                  x           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  #...Compute a Jacobian in  i direction
                  Jf = uxx*(uyy*uzz-uzy*uyz) - uxy*(uxx*uzz - uzx*uyz) +uxz*(uyx*uzy -uzx*uyy)
                  # .... 
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny, nz = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, nz]

                  # Flatten the solution and attach as a scalar field
                  grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
            else:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
                  sy          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
                  sz          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zad[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  # .... 
                  # ... image by analytic function
                  fnc  = Func(x, y, z)
                  # .... 
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny, nz = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, nz]

                  # Flatten the solution and attach as a scalar field
                  grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid

         else:
            if Func is None:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
                  sy          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
                  sz          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zad[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  # .... 
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny, nz = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, nz]

                  # Flatten the solution and attach as a scalar field
                  for sol in solution:
                     # .... 
                     U          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
            else:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xad[i], V[i].knots, V[i].degree)[0]
                  sy          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), yad[i], V[i].knots, V[i].degree)[0]
                  sz          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zad[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  # .... 
                  # ... image bu analytic function
                  fnc  = Func(x, y, z)
                  # .... 
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny, nz = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, nz]

                  # Flatten the solution and attach as a scalar field
                  grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
                  for sol in solution:
                     # .... 
                     U          = pyccel_sol_field_3d((nbpts, nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                     grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid         
   # Save multiblock dataset
   multiblock.save(output_path)
   print(f"Saved all patches with solution to {output_path}")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_SolutionMultipatch(nbpts, V, xmp, ymp, zmp = None, solution = None, Func = None, output_path = "figs/multipatch_solution.vtm", plot = True): 
   """
   Post-processes and exports the solution in the multi-patch domain using Paraview.

   Parameters
   ----------
   nbpts : int
       Number of points per patch direction for evaluation.
   V : list
       List of patch objects containing spline spaces.
   xmp, ymp : list
       Lists of control points for the initial mapping in x and y directions.
   zmp : list, optional
       List of control points for the initial mapping in z direction (for 3D).
   solution : list, optional
       List of solution control points for each patch.
   Func : callable, optional
       Analytic function to evaluate on the mesh (signature depends on dimension).
   output_path : str, optional
       Path to save the output VTM file (default: "figs/multipatch_solution.vtm").
   plot : bool, optional
       If True, enables plotting (not used in this function).

   Returns
   -------
   None
       The function saves the multi-block dataset to the specified output path.
   """
   #---Compute a solution
   numPaches = len(V)
   os.makedirs("figs", exist_ok=True)
   multiblock = pv.MultiBlock()
   if zmp is None:
      if solution is None:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               # ... image bu analytic function
               fnc  = Func(x, y)
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid

      else:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               # ... image bu analytic function
               fnc  = Func(x, y)
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   elif V[0].dim == 3: #.. z is not none 3D case
      if solution is None:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, uxx, uxy, uxz = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0:4]
               y, uyx, uyy, uyz = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0:4]
               z, uzx, uzy, uzz = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0:4]
               #...Compute a Jacobian
               Jf = uxx*(uyy*uzz-uzy*uyz) - uxy*(uxx*uzz - uzx*uyz) +uxz*(uyx*uzy -uzx*uyy)
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0]
               y = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
               z = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0]
               # ... image bu analytic function
               fnc  = Func(x, y, z)
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid

      else:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0]
               y = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
               z = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0]
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U = pyccel_sol_field_3d((nbpts, nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x = pyccel_sol_field_3d((nbpts, nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0]
               y = pyccel_sol_field_3d((nbpts, nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
               z = pyccel_sol_field_3d((nbpts, nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0]
               # ... image bu analytic function
               fnc  = Func(x, y, z)
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U = pyccel_sol_field_3d((nbpts, nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   else: #.. z is not none
      if solution is None:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0:3]
               z, F3x, F3y = pyccel_sol_field_2d((nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0]
               y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
               z = pyccel_sol_field_2d((nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0]
               # ... image bu analytic function
               fnc  = Func(x, y, z)
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid

      else:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0]
               y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
               z = pyccel_sol_field_2d((nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0]
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x = pyccel_sol_field_2d((nbpts, nbpts), xmp[i], V[i].knots, V[i].degree)[0]
               y = pyccel_sol_field_2d((nbpts, nbpts), ymp[i], V[i].knots, V[i].degree)[0]
               z = pyccel_sol_field_2d((nbpts, nbpts), zmp[i], V[i].knots, V[i].degree)[0]
               # ... image bu analytic function
               fnc  = Func(x, y, z)
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               for sol in solution:
                  # .... 
                  U          = pyccel_sol_field_2d((nbpts, nbpts), sol["data"][i], V[i].knots, V[i].degree)[0]
                  grid[sol["name"]] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   # Save multiblock dataset
   multiblock.save(output_path)
   print(f"Saved all patches with solution to {output_path}")



# output_pvd_path = "figs/all_patches.pvd"
# filename_base = "figs/Pmultipatch"
# with open(output_pvd_path, 'w') as f:
#    f.write('<?xml version="1.0"?>\n')
#    f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
#    f.write('  <Collection>\n')

#    for idx, (elF1, elF2) in enumerate(zip(F1, F2)):
#       # Create 3D points (z = 0 for 2D)
#       x, y = elF1, elF2
#       z = np.zeros_like(x)
#       points = np.stack((x, y, z), axis=-1)

#       # Convert to PyVista structured grid
#       nx, ny = x.shape
#       grid = pv.StructuredGrid()
#       grid.points = points.reshape(-1, 3)
#       grid.dimensions = [nx, ny, 1]

#       # Save to .vts file
#       vts_filename = f"{filename_base}_{idx}.vts"
#       grid.save(vts_filename)
#       print(f"Saved patch {idx} to {vts_filename}")

#       rel_path = os.path.basename(vts_filename)
#       f.write(f'    <DataSet timestep="{idx}" group="" part="0" file="{rel_path}"/>\n')

#    f.write('  </Collection>\n')
#    f.write('</VTKFile>\n')