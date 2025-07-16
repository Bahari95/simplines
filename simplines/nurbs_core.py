# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Author: M. BAHARI
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def nurbs_ders_on_quad_grid(ne1:'int', p1:'int', spans_1:'int[:]', basis_1:'float[:,:,:,:]', weights_1:'float[:,:]', points_1:'float[:,:]', knots_1:'float[:]', omega:'float[:]', nders:'int'):
    # Assemble NURBS basis functions and their derivatives at quadrature points for 1D elements.

    # Parameters
    # ----------
    # ne1 : int
    #     Number of elements in the 1D mesh.
    # p1 : int
    #     Degree of the NURBS basis.
    # spans_1 : int[:]
    #     Output array to store the knot span index for each element.
    # basis_1 : float[:,:,:,:]
    #     Output array to store basis functions and their derivatives at quadrature points.
    # weights_1 : float[:,:]
    #     Weights for NURBS basis functions.
    # points_1 : float[:,:]
    #     Quadrature points for each element.
    # knots_1 : float[:]
    #     Knot vector.
    # omega : float[:]
    #     Weights for NURBS basis functions.
    # nders : int
    #     Number of derivatives to compute.

    # Notes
    # -----
    # This function computes the non-zero basis functions and their derivatives at each quadrature point
    # for all elements in a 1D mesh, supporting both NURBS basis.
    # ... sizes
    from numpy import zeros
    from numpy import empty
    # ...
    k1             = weights_1.shape[1]
    #   ---Computes All basis in a new points
    degree         = p1
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
          for g1 in range(0, k1):
                xq = points_1[ie1, g1]
                #~~~~~~~~~~~~~~~
                # Knot index at left/right boundary
                low  = degree
                high = len(knots_1)-1-degree
                # Check if point is exactly on left/right boundary, or outside domain
                if xq <= knots_1[low ]: 
                      span = low
                elif xq >= knots_1[high]: 
                      span = high-1
                else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_1[span] or xq >= knots_1[span+1]:
                      if xq < knots_1[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                # ... 
                spans_1[ie1] = span
                # ...
                ndu[0,0] = 1.0
                for j in range(0,degree):
                    left [j] = xq - knots_1[span-j]
                    right[j] = knots_1[span+1+j] - xq
                    saved    = 0.0
                    for r in range(0,j+1):
                        # compute inverse of knot differences and save them into lower triangular part of ndu
                        ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                        # compute basis functions and save them into upper triangular part of ndu
                        temp       = ndu[r,j] * ndu[j+1,r]
                        ndu[r,j+1] = saved + right[r] * temp
                        saved      = left[j-r] * temp
                    ndu[j+1,j+1] = saved	
               
                # Compute derivatives in 2D output array 'ders'
                ders[0,:] = ndu[:,degree]
                for r in range(0,degree+1):
                    s1 = 0
                    s2 = 1
                    a[0,0] = 1.0
                    for k in range(1,nders+1):
                        d  = 0.0
                        rk = r-k
                        pk = degree-k
                        if r >= k:
                            a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                            d = a[s2,0] * ndu[rk,pk]
                        j1 = 1   if (rk  > -1 ) else -rk
                        j2 = k-1 if (r-1 <= pk) else degree-r
                        for ij in range(j1,j2+1):
                             a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                        for ij in range(j1,j2+1):
                             d += a[s2,ij]* ndu[rk+ij,pk]
                        if r <= pk:
                            a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                            d += a[s2,k] * ndu[r,pk]
                        ders[k,r] = d
                        j  = s1
                        s1 = s2
                        s2 = j
                # ...first compute R1
                ders[0,:]     = ders[0,:] * omega[span-degree:]
                sum_basisx    = sum(ders[0,:])
                basis_1[ie1, :, 0, g1] = ders[0,:]/sum_basisx
                r = degree
                for i_ders in range(1,nders+1):
                    # Multiply derivatives by correct factors
                    ders[i_ders,:] = ders[i_ders,:] * r * omega[span-degree:]
                    basis_1[ie1, :, i_ders, g1] = ders[i_ders,:]/sum_basisx
                    for j_ders in range(0,i_ders):
                        basis_1[ie1, :, i_ders, g1] -= (j_ders+1)*basis_1[ie1, :, j_ders, g1] * sum(ders[i_ders-j_ders,:])/sum_basisx
                    r = r * (degree-i_ders)
def nurbs_ders_on_shared_quad_grid(ne1:'int', p1:'int', spans_1:'int[:,:]', basis_1:'float[:,:,:,:]', weights_1:'float[:,:]', points_1:'float[:,:]', knots_1:'float[:]', omega:'float[:]', nders:'int'):
    # Assemble NURBS basis functions and their derivatives at quadrature points for 1D elements.

    # Parameters
    # ----------
    # ne1 : int
    #     Number of elements in the 1D mesh.
    # p1 : int
    #     Degree of the NURBS basis.
    # spans_1 : int[:]
    #     Output array to store the knot span index for each element.
    # basis_1 : float[:,:,:,:]
    #     Output array to store basis functions and their derivatives at quadrature points.
    # weights_1 : float[:,:]
    #     Weights for NURBS basis functions.
    # points_1 : float[:,:]
    #     Quadrature points for each element.
    # knots_1 : float[:]
    #     Knot vector.
    # omega : float[:]
    #     Weights for NURBS basis functions.
    # nders : int
    #     Number of derivatives to compute.

    # Notes
    # -----
    # This function computes the non-zero basis functions and their derivatives at each quadrature point
    # for all elements in a 1D mesh, supporting both NURBS basis.
    # ... sizes
    from numpy import zeros
    from numpy import empty
    # ...
    k1             = weights_1.shape[1]
    #   ---Computes All basis in a new points
    degree         = p1
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
          for g1 in range(0, k1):
                xq = points_1[ie1, g1]
                #~~~~~~~~~~~~~~~
                # Knot index at left/right boundary
                low  = degree
                high = len(knots_1)-1-degree
                # Check if point is exactly on left/right boundary, or outside domain
                if xq <= knots_1[low ]: 
                      span = low
                elif xq >= knots_1[high]: 
                      span = high-1
                else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_1[span] or xq >= knots_1[span+1]:
                      if xq < knots_1[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                # ... 
                spans_1[ie1, g1] = span
                # ...
                ndu[0,0] = 1.0
                for j in range(0,degree):
                    left [j] = xq - knots_1[span-j]
                    right[j] = knots_1[span+1+j] - xq
                    saved    = 0.0
                    for r in range(0,j+1):
                        # compute inverse of knot differences and save them into lower triangular part of ndu
                        ndu[j+1,r] = 1.0 / (right[r] + left[j-r])
                        # compute basis functions and save them into upper triangular part of ndu
                        temp       = ndu[r,j] * ndu[j+1,r]
                        ndu[r,j+1] = saved + right[r] * temp
                        saved      = left[j-r] * temp
                    ndu[j+1,j+1] = saved	
               
                # Compute derivatives in 2D output array 'ders'
                ders[0,:] = ndu[:,degree]
                for r in range(0,degree+1):
                    s1 = 0
                    s2 = 1
                    a[0,0] = 1.0
                    for k in range(1,nders+1):
                        d  = 0.0
                        rk = r-k
                        pk = degree-k
                        if r >= k:
                            a[s2,0] = a[s1,0] * ndu[pk+1,rk]
                            d = a[s2,0] * ndu[rk,pk]
                        j1 = 1   if (rk  > -1 ) else -rk
                        j2 = k-1 if (r-1 <= pk) else degree-r
                        for ij in range(j1,j2+1):
                             a[s2,ij] = (a[s1,ij] - a[s1,ij-1]) * ndu[pk+1,rk+ij]
                        for ij in range(j1,j2+1):
                             d += a[s2,ij]* ndu[rk+ij,pk]
                        if r <= pk:
                            a[s2,k] = - a[s1,k-1] * ndu[pk+1,r]
                            d += a[s2,k] * ndu[r,pk]
                        ders[k,r] = d
                        j  = s1
                        s1 = s2
                        s2 = j
                # ...first compute R1
                ders[0,:]     = ders[0,:] * omega[span-degree:]
                sum_basisx    = sum(ders[0,:])
                basis_1[ie1, :, 0, g1] = ders[0,:]/sum_basisx
                r = degree
                for i_ders in range(1,nders+1):
                    # Multiply derivatives by correct factors
                    ders[i_ders,:] = ders[i_ders,:] * r * omega[span-degree:]
                    basis_1[ie1, :, i_ders, g1] = ders[i_ders,:]/sum_basisx
                    for j_ders in range(0,i_ders):
                        basis_1[ie1, :, i_ders, g1] -= (j_ders+1)*basis_1[ie1, :, j_ders, g1] * sum(ders[i_ders-j_ders,:])/sum_basisx
                    r = r * (degree-i_ders)