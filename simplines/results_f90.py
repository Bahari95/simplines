from pyccel.decorators import types
from pyccel.epyccel import epyccel

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
@types('int', 'int', 'real[:,:]', 'real[:,:]', 'real[:,:]', 'real[:]', 'real[:]', 'int', 'int', 'real[:,:,:]')
def sol_field_2D_meshes(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, Q):
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
          
              x = xs[i_x,j_y]
              y = ys[i_x,j_y]
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
              

f90_sol_field_2d_meshes = epyccel(sol_field_2D_meshes)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
@types('int', 'int', 'real[:]', 'real[:]', 'real[:,:]', 'real[:]', 'real[:]', 'int', 'int', 'real[:,:,:]')
def sol_field_2D(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, Q):
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
              
f90_sol_field_2d = epyccel(sol_field_2D)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
@types('int', 'int', 'int', 'real[:]', 'real[:]', 'real[:]', 'real[:,:,:]', 'real[:]', 'real[:]', 'real[:]', 'int', 'int', 'int', 'real[:,:,:,:]')
def sol_field_3D(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tw, pu, pv, pw, Q):
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
              
f90_sol_field_3d = epyccel(sol_field_3D)

#==============================================================
from numpy import zeros, linspace, meshgrid
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def pyccel_sol_field_2d( Npoints, uh, knots, degree, meshes = None):
    '''
    Using computed control points uh we compute solution
    in new discretisation by Npoints    
    '''
    pu, pv = degree
    Tu, Tv = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    
    if meshes is None:

	    if Npoints is None:

	       nx = nu-pu+1
	       ny = nv-pv+1
	    
	       xs = Tu[pu:-pu] #linspace(Tu[pu], Tu[-pu-1], nx)
	    
	       ys = Tv[pv:-pv] #linspace(Tv[pv], Tv[-pv-1], ny)
	      
	    else :
	       nx, ny = Npoints

	       xs = linspace(Tu[pu], Tu[-pu-1], nx)
	    
	       ys = linspace(Tv[pv], Tv[-pv-1], ny)

	    Q    = zeros((nx, ny, 3)) 
	    f90_sol_field_2d(nx, ny, xs, ys, uh, Tu, Tv, pu, pv, Q)
	    X, Y = meshgrid(xs, ys)
	    return Q[:,:,0], Q[:,:,1], Q[:,:,2], X.T, Y.T
    else :
       xs, ys = meshes
       #print(xs)
       nx, ny = xs.shape
       Xs     = linspace(Tu[pu], Tu[-pu-1], nx)    
       Ys     = linspace(Tv[pv], Tv[-pv-1], ny)
       X, Y   = meshgrid(Xs, Ys)
       #print(X)
       X[:,:] = xs[:,:]
       Y[:,:] = ys[:,:]
       Q      = zeros((nx, ny, 3)) 
       f90_sol_field_2d_meshes(nx, ny, X, Y, uh, Tu, Tv, pu, pv, Q)       
       return Q[:,:,0], Q[:,:,1], Q[:,:,2]

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Computes Solution and its gradien In two dimension
def pyccel_sol_field_3d(Npoints,  uh , knots, degree):
    # Using computed control points U we compute solution
    # in new discretisation by Npoints    
    pu, pv, pz = degree
    Tu, Tv, Tz = knots

    nu = len(Tu) - pu - 1
    nv = len(Tv) - pv - 1    
    nz = len(Tz) - pz - 1    
    
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
    f90_sol_field_3d(nx, ny, nz, xs, ys, zs, uh, Tu, Tv, Tz, pu, pv, pz, Q)

    return Q[:,:,:,0], Q[:,:,:,1], Q[:,:,:,2], Q[:,:,:,3], Q[:,:,:,4], Q[:,:,:,5], Q[:,:,:,6]
    

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
        #...x_mae is not None implies that the Boudary conditions are fulfilled after applying the optimal mapping to the boundary points
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
