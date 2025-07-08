"""
This module provides utilities for NURBS-based computations, including solution field evaluation,
prolongation, and Paraview post-processing for multi-patch domains.

@author : M. BAHARI
"""
from   .               import nurbs_utilities_core as core

from numpy import zeros, linspace, meshgrid
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_NURBS_2d( Npoints, uh, omega, knots, degree, meshes = None, bound_val = None):
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
        w1, w2 = omega
        Q      = zeros((nx, ny, 3)) 
        core.sol_field_2D(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, w1, w2, Q)
        # ...
        X, Y   = meshgrid(xs, ys)
        return Q[:,:,0], Q[:,:,1], Q[:,:,2], X.T, Y.T
    else :
       w1, w2 = omega
       # ...
       nx, ny   = meshes[0].shape
       Q        = zeros((nx, ny, 5))
       Q[:,:,3] = meshes[0][:,:]
       Q[:,:,4] = meshes[1][:,:] 
       core.sol_field_2D_meshes(nx, ny, uh, Tu, Tv, pu, pv, w1, w2, Q)       
       return Q[:,:,0], Q[:,:,1], Q[:,:,2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_NURBS_3d(Npoints,  uh , omega, knots, degree, meshes = None):
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
       # ...
       w1, w2, w3 = omega
       Q    = zeros((nx, ny, nz, 7)) 
       core.sol_field_3D(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tz, pu, pv, pz, w1, w2, w3, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3], Q[:,:,:,4], Q[:,:,:,5], Q[:,:,:,6],
    else :
       nx, ny, nz = Npoints
       # ...
       w1, w2, w3 = omega
       # ...
       Q          = zeros((nx, ny, nz, 7))
       Q[:,:,:,4] = meshes[0][:,:,:]
       Q[:,:,:,5] = meshes[1][:,:,:]
       Q[:,:,:,6] = meshes[2][:,:,:]
       core.sol_field_3D_mesh(nx, ny, nz, uh, Tu, Tv, Tz, pu, pv, pz, w1, w2, w3, Q)
       return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3]    
       
def prolongate_NURBS_mapping(VH, Vh, w, Cp):
    #.. Prologation by knots insertion matrix
    from   simplines.utilities    import prolongation_matrix

    M  = prolongation_matrix(VH, Vh)
    if Vh.dim == 2 :
        px, py = Cp
        # ... Prolongate the wieghts first
        z  = M.dot(w.reshape(VH.nbasis[0] * VH.nbasis[1])).reshape(Vh.nbasis)

        # ... 
        px = w * px
        py = w * py

        # ...
        Px = (M.dot(px.reshape(VH.nbasis[0] * VH.nbasis[1])).reshape(Vh.nbasis))/z
        Py = (M.dot(py.reshape(VH.nbasis[0] * VH.nbasis[1])).reshape(Vh.nbasis))/z

        return Px, Py, z[:,0], z[0,:]
    else :
        px, py, pz = Cp
        # ... Prolongate the wieghts first
        z  = M.dot(w.reshape(VH.nbasis[0] * VH.nbasis[1]* VH.nbasis[2])).reshape(Vh.nbasis)

        # ... 
        px = w * px
        py = w * py
        pz = w * pz

        # ...
        Px = (M.dot(px.reshape(VH.nbasis[0] * VH.nbasis[1] * VH.nbasis[2])).reshape(Vh.nbasis))/z
        Py = (M.dot(py.reshape(VH.nbasis[0] * VH.nbasis[1] * VH.nbasis[2])).reshape(Vh.nbasis))/z
        Pz = (M.dot(pz.reshape(VH.nbasis[0] * VH.nbasis[1] * VH.nbasis[2])).reshape(Vh.nbasis))/z

        return Px, Py, Pz, z[:,0,0], z[0,:,0], z[0,0,:]

import numpy                        as     np
import pyvista                      as     pv
import os

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
# ... Post processing using Paraview 
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_nurbsAdMeshMultipatch(nbpts, V, xmp, ymp, xad, yad, omega, zad = None, zmp = None, xuh = None, Func = None, output_path = "figs/admultipatch_multiblock.vtm", plot = True): 
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
   omega : list
       List of weights in one direction for each patch
   zad : list, optional
       List of control points for the adaptive mesh in z direction (for 3D).
   zmp : list, optional
       List of control points for the initial mapping in z direction (for 3D). 
   xuh : list, optional
       List of solution control points for each patch.
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
      if xuh is None:
         if Func is None:
            for i in range(numPaches):
               #... computes adaptive meshomega[i], 
               sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
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
               sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
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
               sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               #...Compute a Jacobian
               #Jf = (F1x*F2y - F1y*F2x)
               Jf = (sxx*syy-sxy*syx)*(F1x*F2y - F1y*F2x)
               # .... 
               U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #... computes adaptive mesh
               sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0:3]
               #---Compute a image by initial mapping
               x, F1x, F1y = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
               # .... 
               U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
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
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   else:
      if zad is None :
         if xuh is None:
            if Func is None:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx, sxx, sxy = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0:3]
                  sy, syx, syy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0:3]
                  #---Compute a image by initial mapping
                  x, F1x, F1y = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
                  y, F2x, F2y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0:3]
                  z           = sol_field_NURBS_2d((None, None), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
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
                  sx = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  z = sol_field_NURBS_2d((None, None), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
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
                  sx = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  z = sol_field_NURBS_2d((None, None), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  # .... 
                  U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
                  #...
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, 1]

                  # Flatten the solution and attach as a scalar field
                  grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
            else:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx = sol_field_NURBS_2d((nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sy = sol_field_NURBS_2d((nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x = sol_field_NURBS_2d((None, None), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  y = sol_field_NURBS_2d((None, None), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  z = sol_field_NURBS_2d((None, None), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy))[0]
                  # .... 
                  # .... 
                  U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
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
                  grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
      else: #... zad
         if xuh is None:
            if Func is None:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx, uxx, uxy, uxz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0:4]
                  sy, uyx, uyy, uyz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0:4]
                  sz, uzx, uzy, uzz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zad[i], omega[i], V[i].knots, V[i].degree)[0:4]
                  #---Compute a image by initial mapping
                  x           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
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
                  sx          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sy          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sz          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zad[i], omega[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
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
                  sx          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sy          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sz          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zad[i], omega[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  # .... 
                  U          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
                  # .... 
                  points = np.stack((x, y, z), axis=-1)

                  nx, ny, nz = x.shape
                  grid = pv.StructuredGrid()
                  grid.points = points.reshape(-1, 3)
                  grid.dimensions = [nx, ny, nz]

                  # Flatten the solution and attach as a scalar field
                  grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid
            else:
               for i in range(numPaches):
                  #... computes adaptive mesh
                  sx          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sy          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), yad[i], omega[i], V[i].knots, V[i].degree)[0]
                  sz          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zad[i], omega[i], V[i].knots, V[i].degree)[0]
                  #---Compute a image by initial mapping
                  x           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  y           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  z           = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree, meshes=(sx, sy, sz))[0]
                  # .... 
                  # .... 
                  U          = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
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
                  grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

                  multiblock[f"patch_{i}"] = grid         
   # Save multiblock dataset
   multiblock.save(output_path)
   print(f"Saved all patches with solution to {output_path}")


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def paraview_nurbsSolutionMultipatch(nbpts, V, xmp, ymp, omega, zmp = None, xuh = None, Func = None, output_path = "figs/multipatch_solution.vtm", plot = True): 
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
   omega : list
       List of weights in one direction for each patch
   zmp : list, optional
       List of control points for the initial mapping in z direction (for 3D).
   xuh : list, optional
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
      if xuh is None:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0:3]
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
               x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0:3]
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
               x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               # .... 
               U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
               #...
               z = np.zeros_like(x)
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               grid["Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               #...Compute a Jacobian
               Jf = F1x*F2y - F1y*F2x
               # ... image bu analytic function
               fnc  = Func(x, y)
               # .... 
               U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
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
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   elif V[0].dim == 3: #.. z is not none 3D case
      if xuh is None:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, uxx, uxy, uxz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0:4]
               y, uyx, uyy, uyz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0:4]
               z, uzx, uzy, uzz = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0:4]
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
               x = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0]
               y = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0]
               z = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0]
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
               x = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0]
               y = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0]
               z = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0]
               # .... 
               U = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0]
               y = sol_field_NURBS_3d((nbpts, nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0]
               z = sol_field_NURBS_3d((nbpts, nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0]
               # ... image bu analytic function
               fnc  = Func(x, y, z)
               # .... 
               U = sol_field_NURBS_3d((nbpts, nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
               # .... 
               points = np.stack((x, y, z), axis=-1)

               nx, ny, nz = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, nz]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   else: #.. z is not none
      if xuh is None:
         if Func is None:
            for i in range(numPaches):
               #---Compute a physical domain
               x, F1x, F1y = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               y, F2x, F2y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0:3]
               z, F3x, F3y = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0:3]
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
               x = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0]
               y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0]
               z = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0]
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
               x = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0]
               y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0]
               z = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0]
               # .... 
               U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
         else:
            for i in range(numPaches):
               #---Compute a physical domain
               x = sol_field_NURBS_2d((nbpts, nbpts), xmp[i], omega[i], V[i].knots, V[i].degree)[0]
               y = sol_field_NURBS_2d((nbpts, nbpts), ymp[i], omega[i], V[i].knots, V[i].degree)[0]
               z = sol_field_NURBS_2d((nbpts, nbpts), zmp[i], omega[i], V[i].knots, V[i].degree)[0]
               # ... image bu analytic function
               fnc  = Func(x, y, z)
               # .... 
               U          = sol_field_NURBS_2d((nbpts, nbpts), xuh[i], omega[i], V[i].knots, V[i].degree)[0]
               #...
               points = np.stack((x, y, z), axis=-1)

               nx, ny = x.shape
               grid = pv.StructuredGrid()
               grid.points = points.reshape(-1, 3)
               grid.dimensions = [nx, ny, 1]

               # Flatten the solution and attach as a scalar field
               # grid["i-Jacobain"] = Jf.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["Analytic Function"] = fnc.flatten(order='C')  # or 'F' if needed (check your ordering)
               grid["solution"] = U.flatten(order='C')  # or 'F' if needed (check your ordering)

               multiblock[f"patch_{i}"] = grid
   # Save multiblock dataset
   multiblock.save(output_path)
   print(f"Saved all patches with solution to {output_path}")