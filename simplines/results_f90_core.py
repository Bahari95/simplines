#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""
#Parameters
----------
nx : int
    Number of points in the mesh.
uh : float[:]
    Array of control point values.
Tu : float[:]
    Knot vector.
pu : int
    Degree of the basis functions.
Q : float[:,:]
    Output array where the computed solution and its gradient will be stored.
    Q[:, 0] will contain the solution values, Q[:, 1] will contain the gradients.

Notes
-----
This function evaluates the B-spline solution and its derivative at the given mesh points.
"""
def sol_field_1D_meshes(nx:'int', uh:'float[:]', Tu:'float[:]', pu:'int', Q:'float[:,:]'):
    #Computes the solution and its gradient at each point in a 1D mesh.
    
    from numpy import zeros
    from numpy import empty

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #...
    for i_x in range(nx):          
              x = Q[i_x,2]
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span     
              #...
              bu      = basis_x[0,:]    
              derbu   = basis_x[1,:]
          
              c       = 0.
              cx      = 0.
              for ku in range(0, pu+1):
                    c  += bu[ku]*uh[span_u-pu+ku]
                    cx += derbu[ku]*uh[span_u-pu+ku]
              #..
              Q[i_x, 0]   = c
              Q[i_x, 1]   = cx
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_2D_meshes(nx:'int', ny:'int', uh:'float[:,:]', Tu:'float[:]', Tv:'float[:]', pu:'int', pv:'int', Q:'float[:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = zeros((nu, nv))
    
    for i in range(nu):  
       for j in range(nv):
             P[i, j] = uh[i, j]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          
              x = Q[i_x,j_y,3]
              y = Q[i_x,j_y,4]
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #...
              bu      = basis_x[0,:]
              bv      = basis_y[0,:]
    
              derbu   = basis_x[1,:]
              derbv   = basis_y[1,:]
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                      c  += bu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cx += derbu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cy += bu[ku]*derbv[kv]*P[span_u-pu+ku, span_v-pv+kv]
              #..
              Q[i_x, j_y, 0]   = c
              Q[i_x, j_y, 1]   = cx
              Q[i_x, j_y, 2]   = cy
              
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_2D(nx:'int', ny:'int', xs:'float[:]', ys:'float[:]', uh:'float[:,:]', Tu:'float[:]', Tv:'float[:]', pu:'int', pv:'int', Q:'float[:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1

    P = zeros((nu, nv))
    
    for i in range(nu):  
       for j in range(nv):
             P[i, j] = uh[i, j]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          
              x = xs[i_x]
              y = ys[j_y]
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #...
              bu      = basis_x[0,:]
              bv      = basis_y[0,:]
    
              derbu   = basis_x[1,:]
              derbv   = basis_y[1,:]
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                      c  += bu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cx += derbu[ku]*bv[kv]*P[span_u-pu+ku, span_v-pv+kv]
                      cy += bu[ku]*derbv[kv]*P[span_u-pu+ku, span_v-pv+kv]
              #..
              Q[i_x, j_y, 0]   = c
              Q[i_x, j_y, 1]   = cx
              Q[i_x, j_y, 2]   = cy
              
              
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def sol_field_3D(nx:'int', ny:'int', nz:'int', xs:'float[:]', ys:'float[:]', zs:'float[:]', uh:'float[:,:,:]', Tu:'float[:]', Tv:'float[:]', Tw:'float[:]', pu:'int', pv:'int', pw:'int', Q:'float[:,:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty
    from numpy import linspace

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    nw = len(Tw) - pw - 1

    P = zeros((nu, nv, nw))
    
    for i in range(nu):  
       for j in range(nv):
          for k in range(nw):
             P[i, j, k] = uh[i, j, k]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    #
    leftw      = empty( pw )
    rightw     = empty( pw )
    ndw        = empty( (pw+1, pw+1) )
    aw         = empty( (       2, pw+1) )
    dersw      = zeros( (     nders+1, pw+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          for k_z in range(nz):
          
              x = xs[i_x]
              y = ys[j_y]
              z = zs[k_z]    
              
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #basis_z = basis_funs_all_ders( Tw, pw, z, span_w, 1 )
              #--Computes All basis in a new points
              xq         = z
              dersw[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pw
              high = len(Tw)-1-pw
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tw[low ]: 
                   span = low
              elif xq >= Tw[high]: 
                  span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tw[span] or xq >= Tw[span+1]:
                   if xq < Tw[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndw[0,0] = 1.0
              for j in range(0,pw):
                  leftw [j] = xq - Tw[span-j]
                  rightw[j] = Tw[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndw[j+1,r] = 1.0 / (rightw[r] + leftw[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndw[r,j] * ndw[j+1,r]
                      ndw[r,j+1] = saved + rightw[r] * temp
                      saved      = leftw[j-r] * temp
                  ndw[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersw[0,:] = ndw[:,pw]
              for r in range(0,pw+1):
                  s1 = 0
                  s2 = 1
                  aw[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pw-k
                      if r >= k:
                         aw[s2,0] = aw[s1,0] * ndw[pk+1,rk]
                         d = aw[s2,0] * ndw[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pw-r
                      for ij in range(j1,j2+1):
                          aw[s2,ij] = (aw[s1,ij] - aw[s1,ij-1]) * ndw[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += aw[s2,ij]* ndw[rk+ij,pk]
                      if r <= pk:
                         aw[s2,k] = - aw[s1,k-1] * ndw[pk+1,r]
                         d += aw[s2,k] * ndw[r,pk]
                      dersw[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pw
              dersw[1,:] = dersw[1,:] * r
              basis_z = dersw
              span_w  = span
              #...
              bu      = basis_x[0,:]
              bv      = basis_y[0,:]
              bw      = basis_z[0,:]
    
              derbu   = basis_x[1,:]
              derbv   = basis_y[1,:]
              derbw   = basis_z[1,:]
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              cz      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                    for kw in range(0, pw+1):
                      c  += bu[ku]*bv[kv]*bw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cx += derbu[ku]*bv[kv]*bw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cy += bu[ku]*derbv[kv]*bw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cz += bu[ku]*bv[kv]*derbw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
              #..
              Q[i_x, j_y, k_z,0]   = c
              Q[i_x, j_y, k_z,1]   = cx
              Q[i_x, j_y, k_z,2]   = cy
              Q[i_x, j_y, k_z,3]   = cz
              Q[i_x, j_y, k_z,4]   = x
              Q[i_x, j_y, k_z,5]   = y
              Q[i_x, j_y, k_z,6]   = z
              
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In three dimension
def sol_field_3D_mesh(nx:'int', ny:'int', nz:'int', uh:'float[:,:,:]', Tu:'float[:]', Tv:'float[:]', Tw:'float[:]', pu:'int', pv:'int', pw:'int', Q:'float[:,:,:,:]'):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints
    
    from numpy import zeros
    from numpy import empty
    from numpy import linspace

    #pu, pv, pw = p1, p2, p3
    #nx, ny, nz = 50
    #Tu, Tv, Tw = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1
    nw = len(Tw) - pw - 1

    P = zeros((nu, nv, nw))
    
    for i in range(nu):  
       for j in range(nv):
          for k in range(nw):
             P[i, j, k] = uh[i, j, k]    

    nders      = 1
    # ...
    leftu      = empty( pu )
    rightu     = empty( pu )
    ndu        = empty( (pu+1, pu+1) )
    au         = empty( (       2, pu+1) )
    dersu      = zeros( (     nders+1, pu+1) ) 
    #..              
    leftv      = empty( pv )
    rightv     = empty( pv )
    ndv        = empty( (pv+1, pv+1) )
    av         = empty( (       2, pv+1) )
    dersv      = zeros( (     nders+1, pv+1) ) 
    #
    leftw      = empty( pw )
    rightw     = empty( pw )
    ndw        = empty( (pw+1, pw+1) )
    aw         = empty( (       2, pw+1) )
    dersw      = zeros( (     nders+1, pw+1) ) 
    
    #...
    for i_x in range(nx):
       for j_y in range(ny):
          for k_z in range(nz):
          
              x = Q[i_x, j_y, k_z,4]
              y = Q[i_x, j_y, k_z,5]
              z = Q[i_x, j_y, k_z,6]   
              
              #basis_x = basis_funs_all_ders( Tu, pu, x, span_u, 1 )
              # ... for x ----
              #--Computes All basis in a new points              
              #..
              xq         = x
              dersu[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pu
              high = len(Tu)-1-pu
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tu[low ]: 
                   span = low
              elif xq >= Tu[high]: 
                  span = high-1
              else : 
                # Perform binary search
                span = (low+high)//2
                while xq < Tu[span] or xq >= Tu[span+1]:
                   if xq < Tu[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2        
              ndu[0,0] = 1.0
              for j in range(0,pu):
                  leftu [j] = xq - Tu[span-j]
                  rightu[j] = Tu[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndu[j+1,r] = 1.0 / (rightu[r] + leftu[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndu[r,j] * ndu[j+1,r]
                      ndu[r,j+1] = saved + rightu[r] * temp
                      saved      = leftu[j-r] * temp
                  ndu[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersu[0,:] = ndu[:,pu]
              for r in range(0,pu+1):
                  s1 = 0
                  s2 = 1
                  au[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pu-k
                      if r >= k:
                         au[s2,0] = au[s1,0] * ndu[pk+1,rk]
                         d = au[s2,0] * ndu[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pu-r
                      for ij in range(j1,j2+1):
                          au[s2,ij] = (au[s1,ij] - au[s1,ij-1]) * ndu[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += au[s2,ij]* ndu[rk+ij,pk]
                      if r <= pk:
                         au[s2,k] = - au[s1,k-1] * ndu[pk+1,r]
                         d += au[s2,k] * ndu[r,pk]
                      dersu[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pu
              dersu[1,:] = dersu[1,:] * r
              basis_x = dersu
              span_u  = span
              #...
              #basis_y = basis_funs_all_ders( Tv, pv, y, span_v, 1 )
              # ... for y ----
              #--Computes All basis in a new points
              xq         = y
              dersv[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pv
              high = len(Tv)-1-pv
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tv[low ]: 
                   span = low
              elif xq >= Tv[high]: 
                   span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tv[span] or xq >= Tv[span+1]:
                   if xq < Tv[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndv[0,0] = 1.0
              for j in range(0,pv):
                  leftv [j] = xq - Tv[span-j]
                  rightv[j] = Tv[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndv[j+1,r] = 1.0 / (rightv[r] + leftv[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndv[r,j] * ndv[j+1,r]
                      ndv[r,j+1] = saved + rightv[r] * temp
                      saved      = leftv[j-r] * temp
                  ndv[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersv[0,:] = ndv[:,pv]
              for r in range(0,pv+1):
                  s1 = 0
                  s2 = 1
                  av[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pv-k
                      if r >= k:
                         av[s2,0] = av[s1,0] * ndv[pk+1,rk]
                         d = av[s2,0] * ndv[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pv-r
                      for ij in range(j1,j2+1):
                          av[s2,ij] = (av[s1,ij] - av[s1,ij-1]) * ndv[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += av[s2,ij]* ndv[rk+ij,pk]
                      if r <= pk:
                         av[s2,k] = - av[s1,k-1] * ndv[pk+1,r]
                         d += av[s2,k] * ndv[r,pk]
                      dersv[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pv
              dersv[1,:] = dersv[1,:] * r
              basis_y = dersv
              span_v  = span             
              #basis_z = basis_funs_all_ders( Tw, pw, z, span_w, 1 )
              #--Computes All basis in a new points
              xq         = z
              dersw[:,:] = 0.
              #~~~~~~~~~~~~~~~
              # Knot index at left/right boundary
              low  = pw
              high = len(Tw)-1-pw
              # Check if point is exactly on left/right boundary, or outside domain
              if xq <= Tw[low ]: 
                   span = low
              elif xq >= Tw[high]: 
                  span = high-1
              else :
                # Perform binary search
                span = (low+high)//2
                while xq < Tw[span] or xq >= Tw[span+1]:
                   if xq < Tw[span]:
                       high = span
                   else:
                       low  = span
                   span = (low+high)//2              
              ndw[0,0] = 1.0
              for j in range(0,pw):
                  leftw [j] = xq - Tw[span-j]
                  rightw[j] = Tw[span+1+j] - xq
                  saved    = 0.0
                  for r in range(0,j+1):
                      # compute inverse of knot differences and save them into lower triangular part of ndu
                      ndw[j+1,r] = 1.0 / (rightw[r] + leftw[j-r])
                      # compute basis functions and save them into upper triangular part of ndu
                      temp       = ndw[r,j] * ndw[j+1,r]
                      ndw[r,j+1] = saved + rightw[r] * temp
                      saved      = leftw[j-r] * temp
                  ndw[j+1,j+1] = saved	

              # Compute derivatives in 2D output array 'ders'
              dersw[0,:] = ndw[:,pw]
              for r in range(0,pw+1):
                  s1 = 0
                  s2 = 1
                  aw[0,0] = 1.0
                  for k in range(1,nders+1):
                      d  = 0.0
                      rk = r-k
                      pk = pw-k
                      if r >= k:
                         aw[s2,0] = aw[s1,0] * ndw[pk+1,rk]
                         d = aw[s2,0] * ndw[rk,pk]
                      j1 = 1   if (rk  > -1 ) else -rk
                      j2 = k-1 if (r-1 <= pk) else pw-r
                      for ij in range(j1,j2+1):
                          aw[s2,ij] = (aw[s1,ij] - aw[s1,ij-1]) * ndw[pk+1,rk+ij]
                      for ij in range(j1,j2+1):
                          d += aw[s2,ij]* ndw[rk+ij,pk]
                      if r <= pk:
                         aw[s2,k] = - aw[s1,k-1] * ndw[pk+1,r]
                         d += aw[s2,k] * ndw[r,pk]
                      dersw[k,r] = d
                      j  = s1
                      s1 = s2
                      s2 = j
              # Multiply derivatives by correct factors
              r = pw
              dersw[1,:] = dersw[1,:] * r
              basis_z = dersw
              span_w  = span
              #...
              bu      = basis_x[0,:]
              bv      = basis_y[0,:]
              bw      = basis_z[0,:]
    
              derbu   = basis_x[1,:]
              derbv   = basis_y[1,:]
              derbw   = basis_z[1,:]
          
              c       = 0.
              cx      = 0.
              cy      = 0.
              cz      = 0.
              for ku in range(0, pu+1):
                  for kv in range(0, pv+1):
                    for kw in range(0, pw+1):
                      c  += bu[ku]*bv[kv]*bw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cx += derbu[ku]*bv[kv]*bw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cy += bu[ku]*derbv[kv]*bw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
                      cz += bu[ku]*bv[kv]*derbw[kw]*P[span_u-pu+ku, span_v-pv+kv, span_w-pw+kw]
              #..
              Q[i_x, j_y, k_z,0]   = c
              Q[i_x, j_y, k_z,1]   = cx
              Q[i_x, j_y, k_z,2]   = cy
              Q[i_x, j_y, k_z,3]   = cz
