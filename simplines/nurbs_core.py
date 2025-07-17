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
    #     Output vector to store the knot span index for each element.
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
    # for all elements in a 1D mesh, supporting NURBS basis.
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
    #     Output vector to store the knot span index for each element.
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
    # for all elements in a 1D mesh, supporting NURBS basis.
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

#==============================================================================
# ... L2-B-spline space for L^2mapping is the same as for initial mapping.
#==============================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
def assemble_nurbsbasis_spans_in_adquadrature_1DL2map(ne1:'int', p1:'int', spans_1:'int[:]', basis_1:'float[:,:,:,:]', weights_1:'float[:,:]', points_1:'float[:,:]', knots_1:'float[:]', omega:'float[:]', vector_u:'float[:]',  spans_ad:'int[:,:]', basis_ad:'float[:,:,:,:]', nders:'int'):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    from numpy import empty
    # ...
    k1         = weights_1.shape[1]
    # ___
    lcoeffs_u  = zeros(p1+1)
    # ... Initialization
    points1    = zeros((ne1, k1))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]

        lcoeffs_u[:] = vector_u[i_span_1 : i_span_1+p1+1]
        for g1 in range(0, k1):
            sx = 0.0
            for il_1 in range(0, p1+1):
                bj_0     = basis_1[ie1,il_1,0,g1]

                coeff_u1 = lcoeffs_u[il_1]
                sx      += coeff_u1 * bj_0
                        
            points1[ie1, g1] = sx

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
                xq = points1[ie1, g1]

                #span = find_span( knots, degree, xq )
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
                spans_ad[ie1, g1] = span
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
                basis_ad[ie1, :, 0, g1] = ders[0,:]/sum_basisx
                r = degree
                for i_ders in range(1,nders+1):
                    # Multiply derivatives by correct factors
                    ders[i_ders,:] = ders[i_ders,:] * r * omega[span-degree:]
                    basis_ad[ie1, :, i_ders, g1] = ders[i_ders,:]/sum_basisx
                    for j_ders in range(0,i_ders):
                        basis_ad[ie1, :, i_ders, g1] -= (j_ders+1)*basis_ad[ie1, :, j_ders, g1] * sum(ders[i_ders-j_ders,:])/sum_basisx
                    r = r * (degree-i_ders)
#==============================================================================
# ... L2-B-spline space for L^2 mapping is the same as for initial mapping.
#==============================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
def assemble_nurbsbasis_spans_in_adquadrature_L2map(ne1:'int', ne2:'int', p1:'int', p2:'int', spans_1:'int[:]', spans_2:'int[:]', basis_1:'float[:,:,:,:]', basis_2:'float[:,:,:,:]', weights_1:'float[:,:]', weights_2:'float[:,:]', points_1:'float[:,:]', points_2:'float[:,:]', knots_1:'float[:]', knots_2:'float[:]', omega_1:'float[:]', omega_2:'float[:]', vector_u1:'float[:,:]', vector_u2:'float[:,:]', spans_ad1:'int[:,:,:,:]', spans_ad2:'int[:,:,:,:]', basis_ad1:'float[:,:,:,:,:,:]', basis_ad2:'float[:,:,:,:,:,:]', nders:'int'):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    from numpy import empty
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ___
    lcoeffs_u1   = zeros((p1+1,p2+1))
    lcoeffs_u2   = zeros((p1+1,p2+1))
    # ... Initialization
    points1    = zeros((ne1, ne2, k1, k2))
    points2    = zeros((ne1, ne2, k1, k2))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_u1[ : , : ] = vector_u1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u2[ : , : ] = vector_u2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]

                              coeff_u1 = lcoeffs_u1[il_1,il_2]
                              coeff_u2 = lcoeffs_u2[il_1,il_2]

                              sx     += coeff_u1 * bj_0
                              sy     += coeff_u2 * bj_0
                              
                    points1[ie1, ie2, g1, g2] = sx
                    points2[ie1, ie2, g1, g2] = sy

    #   ---Computes All basis in a new points
    degree         = p1
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points1[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
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
                 spans_ad1[ie1, ie2, g1, g2] = span
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
                 ders[0,:]     = ders[0,:] * omega_1[span-degree:]
                 sum_basisx    = sum(ders[0,:])
                 basis_ad1[ie1, ie2, :, 0, g1, g2] = ders[0,:]/sum_basisx
                 r = degree
                 for i_ders in range(1,nders+1):
                    # Multiply derivatives by correct factors
                    ders[i_ders,:] = ders[i_ders,:] * r * omega_1[span-degree:]
                    basis_ad1[ie1, ie2,  :, i_ders, g1, g2] = ders[i_ders,:]/sum_basisx
                    for j_ders in range(0,i_ders):
                        basis_ad1[ie1, ie2,  :, i_ders, g1, g2] -= (j_ders+1)*basis_ad1[ie1, ie2,  :, j_ders, g1, g2] * sum(ders[i_ders-j_ders,:])/sum_basisx
                    r = r * (degree-i_ders)
    degree         = p2
    # ...
    left2          = empty( degree )
    right2         = empty( degree )
    a2             = empty( (       2, degree+1) )
    ndu2           = empty( (degree+1, degree+1) )
    ders2          = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points2[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left2/right2 boundary
                 low  = degree
                 high = len(knots_2)-1-degree
                 # Check if point is exactly on left2/right2 boundary, or outside domain
                 if xq <= knots_2[low ]: 
                      span = low
                 elif xq >= knots_2[high]: 
                      span = high-1
                 else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_2[span] or xq >= knots_2[span+1]:
                      if xq < knots_2[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                 # ... 
                 spans_ad2[ie1, ie2, g1, g2] = span
                 # ...
                 ndu2[0,0] = 1.0
                 for j in range(0,degree):
                     left2 [j] = xq - knots_2[span-j]
                     right2[j] = knots_2[span+1+j] - xq
                     saved    = 0.0
                     for r in range(0,j+1):
                         # compute inverse of knot differences and save them into lower triangular part of ndu2
                         ndu2[j+1,r] = 1.0 / (right2[r] + left2[j-r])
                         # compute basis functions and save them into upper triangular part of ndu2
                         temp       = ndu2[r,j] * ndu2[j+1,r]
                         ndu2[r,j+1] = saved + right2[r] * temp
                         saved      = left2[j-r] * temp
                     ndu2[j+1,j+1] = saved	
               
                 # Compute derivatives in 2D output array 'ders2'
                 ders2[0,:] = ndu2[:,degree]
                 for r in range(0,degree+1):
                     s1 = 0
                     s2 = 1
                     a2[0,0] = 1.0
                     for k in range(1,nders+1):
                         d  = 0.0
                         rk = r-k
                         pk = degree-k
                         if r >= k:
                            a2[s2,0] = a2[s1,0] * ndu2[pk+1,rk]
                            d = a2[s2,0] * ndu2[rk,pk]
                         j1 = 1   if (rk  > -1 ) else -rk
                         j2 = k-1 if (r-1 <= pk) else degree-r
                         for ij in range(j1,j2+1):
                             a2[s2,ij] = (a2[s1,ij] - a2[s1,ij-1]) * ndu2[pk+1,rk+ij]
                         for ij in range(j1,j2+1):
                             d += a2[s2,ij]* ndu2[rk+ij,pk]
                         if r <= pk:
                            a2[s2,k] = - a2[s1,k-1] * ndu2[pk+1,r]
                            d += a2[s2,k] * ndu2[r,pk]
                         ders2[k,r] = d
                         j  = s1
                         s1 = s2
                         s2 = j
                 # ...first compute R1
                 ders2[0,:]     = ders2[0,:] * omega_2[span-degree:]
                 sum_basisx    = sum(ders2[0,:])
                 basis_ad2[ie1, ie2, :, 0, g1, g2] = ders2[0,:]/sum_basisx
                 r = degree
                 for i_ders in range(1,nders+1):
                    # Multiply derivatives by correct factors
                    ders2[i_ders,:] = ders2[i_ders,:] * r * omega_2[span-degree:]
                    basis_ad2[ie1, ie2,  :, i_ders, g1, g2] = ders2[i_ders,:]/sum_basisx
                    for j_ders in range(0,i_ders):
                        basis_ad2[ie1, ie2,  :, i_ders, g1, g2] -= (j_ders+1)*basis_ad2[ie1, ie2,  :, j_ders, g1, g2] * sum(ders2[i_ders-j_ders,:])/sum_basisx
                    r = r * (degree-i_ders)
#==============================================================================
# ... L2-B-spline space for L^2mapping is the same as for initial mapping. in 3D
#==============================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
def assemble_nurbsbasis_spans_in_adquadrature_3L2map(ne1:'int', ne2:'int', ne3:'int', p1:'int', p2:'int', p3:'int', spans_1:'int[:]', spans_2:'int[:]', spans_3:'int[:]',  basis_1:'float[:,:,:,:]', basis_2:'float[:,:,:,:]', basis_3:'float[:,:,:,:]',  weights_1:'float[:,:]', weights_2:'float[:,:]', weights_3:'float[:,:]', points_1:'float[:,:]', points_2:'float[:,:]', points_3:'float[:,:]', knots_1:'float[:]', knots_2:'float[:]', knots_3:'float[:]', omega_1:'float[:]', omega_2:'float[:]', omega_3:'float[:]', vector_u1:'float[:,:,:]', vector_u2:'float[:,:,:]', vector_u3:'float[:,:,:]', spans_ad1:'int[:,:,:,:,:,:]', spans_ad2:'int[:,:,:,:,:,:]', spans_ad3:'int[:,:,:,:,:,:]', basis_ad1:'float[:,:,:,:,:,:,:,:]', basis_ad2:'float[:,:,:,:,:,:,:,:]', basis_ad3:'float[:,:,:,:,:,:,:,:]', nders:'int'):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    from numpy import empty
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    k3 = weights_3.shape[1]
    # ___
    lcoeffs_u1   = zeros((p1+1,p2+1,p3+1))
    lcoeffs_u2   = zeros((p1+1,p2+1,p3+1))
    lcoeffs_u3   = zeros((p1+1,p2+1,p3+1))
    # ... Initialization
    points1      = zeros((ne1, ne2, ne3, k1, k2, k3))
    points2      = zeros((ne1, ne2, ne3, k1, k2, k3))
    points3      = zeros((ne1, ne2, ne3, k1, k2, k3))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            for ie3 in range(0, ne3):
                i_span_3 = spans_3[ie3]

                lcoeffs_u1[ : , : , : ]  =  vector_u1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
                lcoeffs_u2[ : , : , : ]  =  vector_u2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
                lcoeffs_u3[ : , : , : ]  =  vector_u3[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
                for g1 in range(0, k1):
                   for g2 in range(0, k2):
                      for g3 in range(0, k3):

                        sx = 0.0
                        sy = 0.0
                        sz = 0.0
                        for il_1 in range(0, p1+1):
                            for il_2 in range(0, p2+1):
                                for il_3 in range(0, p3+1):
                                    bj_0     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]*basis_3[ie3,il_3,0,g3]

                                    coeff_u1 = lcoeffs_u1[il_1,il_2,il_3]
                                    coeff_u2 = lcoeffs_u2[il_1,il_2,il_3]
                                    coeff_u3 = lcoeffs_u3[il_1,il_2,il_3]

                                    sx      += coeff_u1 * bj_0
                                    sy      += coeff_u2 * bj_0
                                    sz      += coeff_u3 * bj_0
                        points1[ie1, ie2, ie3, g1, g2, g3] = sx
                        points2[ie1, ie2, ie3, g1, g2, g3] = sy
                        points3[ie1, ie2, ie3, g1, g2, g3] = sz

    #   ---Computes All basis in a new points
    degree         = p1
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
            for ie3 in range(0, ne3):
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        for g3 in range(0, k3):
                            xq = points1[ie1, ie2, ie3, g1, g2, g3]

                            #span = find_span( knots, degree, xq )
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
                            spans_ad1[ie1, ie2, ie3, g1, g2, g3] = span
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
                            ders[0,:]     = ders[0,:] * omega_1[span-degree:]
                            sum_basisx    = sum(ders[0,:])
                            basis_ad1[ie1, ie2, ie3, :, 0, g1, g2, g3] = ders[0,:]/sum_basisx
                            r = degree
                            for i_ders in range(1,nders+1):
                                # Multiply derivatives by correct factors
                                ders[i_ders,:] = ders[i_ders,:] * r * omega_1[span-degree:]
                                basis_ad1[ie1, ie2, ie3, :, i_ders, g1, g2, g3] = ders[i_ders,:]/sum_basisx
                                for j_ders in range(0,i_ders):
                                    basis_ad1[ie1, ie2, ie3, :, i_ders, g1, g2, g3] -= (j_ders+1)*basis_ad1[ie1, ie2, ie3, :, j_ders, g1, g2, g3] * sum(ders[i_ders-j_ders,:])/sum_basisx
                                r = r * (degree-i_ders)
    degree         = p2
    #...
    left2          = empty( degree )
    right2         = empty( degree )
    a2             = empty( (       2, degree+1) )
    ndu2           = empty( (degree+1, degree+1) )
    ders2          = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
           for ie3 in range(0, ne3):
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        for g3 in range(0, k3):
                            xq = points2[ie1, ie2, ie3, g1, g2, g3]

                            #span = find_span( knots, degree, xq )
                            #~~~~~~~~~~~~~~~
                            # Knot index at left2/right2 boundary
                            low  = degree
                            high = len(knots_2)-1-degree
                            # Check if point is exactly on left2/right2 boundary, or outside domain
                            if xq <= knots_2[low ]: 
                                span = low
                            elif xq >= knots_2[high]: 
                                span = high-1
                            else : 
                                # Perform binary search
                                span = (low+high)//2
                                while xq < knots_2[span] or xq >= knots_2[span+1]:
                                    if xq < knots_2[span]:
                                        high = span
                                    else:
                                        low  = span
                                    span = (low+high)//2
                            # ... 
                            spans_ad2[ie1, ie2, ie3, g1, g2, g3] = span
                            # ...
                            ndu2[0,0] = 1.0
                            for j in range(0,degree):
                                left2 [j] = xq - knots_2[span-j]
                                right2[j] = knots_2[span+1+j] - xq
                                saved    = 0.0
                                for r in range(0,j+1):
                                    # compute inverse of knot differences and save them into lower triangular part of ndu2
                                    ndu2[j+1,r] = 1.0 / (right2[r] + left2[j-r])
                                    # compute basis functions and save them into upper triangular part of ndu2
                                    temp       = ndu2[r,j] * ndu2[j+1,r]
                                    ndu2[r,j+1] = saved + right2[r] * temp
                                    saved      = left2[j-r] * temp
                                ndu2[j+1,j+1] = saved	
                        
                            # Compute derivatives in 2D output array 'ders2'
                            ders2[0,:] = ndu2[:,degree]
                            for r in range(0,degree+1):
                                s1 = 0
                                s2 = 1
                                a2[0,0] = 1.0
                                for k in range(1,nders+1):
                                    d  = 0.0
                                    rk = r-k
                                    pk = degree-k
                                    if r >= k:
                                        a2[s2,0] = a2[s1,0] * ndu2[pk+1,rk]
                                        d = a2[s2,0] * ndu2[rk,pk]
                                    j1 = 1   if (rk  > -1 ) else -rk
                                    j2 = k-1 if (r-1 <= pk) else degree-r
                                    for ij in range(j1,j2+1):
                                        a2[s2,ij] = (a2[s1,ij] - a2[s1,ij-1]) * ndu2[pk+1,rk+ij]
                                    for ij in range(j1,j2+1):
                                        d += a2[s2,ij]* ndu2[rk+ij,pk]
                                    if r <= pk:
                                        a2[s2,k] = - a2[s1,k-1] * ndu2[pk+1,r]
                                        d += a2[s2,k] * ndu2[r,pk]
                                    ders2[k,r] = d
                                    j  = s1
                                    s1 = s2
                                    s2 = j
                            # ...first compute R1
                            ders2[0,:]     = ders2[0,:] * omega_2[span-degree:]
                            sum_basisx    = sum(ders2[0,:])
                            basis_ad2[ie1, ie2, ie3, :, 0, g1, g2, g3] = ders2[0,:]/sum_basisx
                            r = degree
                            for i_ders in range(1,nders+1):
                                # Multiply derivatives by correct factors
                                ders2[i_ders,:] = ders2[i_ders,:] * r * omega_2[span-degree:]
                                basis_ad2[ie1, ie2, ie3, :, i_ders, g1, g2, g3] = ders2[i_ders,:]/sum_basisx
                                for j_ders in range(0,i_ders):
                                    basis_ad2[ie1, ie2, ie3, :, i_ders, g1, g2, g3] -= (j_ders+1)*basis_ad2[ie1, ie2, ie3, :, j_ders, g1, g2, g3] * sum(ders2[i_ders-j_ders,:])/sum_basisx
                                r = r * (degree-i_ders)
    degree         = p3
    #...
    left3          = empty( degree )
    right3         = empty( degree )
    a3             = empty( (       2, degree+1) )
    ndu3           = empty( (degree+1, degree+1) )
    ders3          = zeros( (     nders+1, degree+1) ) # output array
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
            for ie3 in range(0, ne3):
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        for g3 in range(0, k3):
                            xq = points2[ie1, ie2, ie3, g1, g2, g3]

                            #span = find_span( knots, degree, xq )
                            #~~~~~~~~~~~~~~~
                            # Knot index at left3/right3 boundary
                            low  = degree
                            high = len(knots_3)-1-degree
                            # Check if point is exactly on left3/right3 boundary, or outside domain
                            if xq <= knots_3[low ]: 
                                span = low
                            elif xq >= knots_3[high]: 
                                span = high-1
                            else : 
                                # Perform binary search
                                span = (low+high)//2
                                while xq < knots_3[span] or xq >= knots_3[span+1]:
                                    if xq < knots_3[span]:
                                        high = span
                                    else:
                                        low  = span
                                    span = (low+high)//2
                            # ... 
                            spans_ad3[ie1, ie2, ie3, g1, g2, g3] = span
                            # ...
                            ndu3[0,0] = 1.0
                            for j in range(0,degree):
                                left3 [j] = xq - knots_3[span-j]
                                right3[j] = knots_3[span+1+j] - xq
                                saved    = 0.0
                                for r in range(0,j+1):
                                    # compute inverse of knot differences and save them into lower triangular part of ndu3
                                    ndu3[j+1,r] = 1.0 / (right3[r] + left3[j-r])
                                    # compute basis functions and save them into upper triangular part of ndu3
                                    temp       = ndu3[r,j] * ndu3[j+1,r]
                                    ndu3[r,j+1] = saved + right3[r] * temp
                                    saved      = left3[j-r] * temp
                                ndu3[j+1,j+1] = saved	
                        
                            # Compute derivatives in 2D output array 'ders3'
                            ders3[0,:] = ndu3[:,degree]
                            for r in range(0,degree+1):
                                s1 = 0
                                s2 = 1
                                a3[0,0] = 1.0
                                for k in range(1,nders+1):
                                    d  = 0.0
                                    rk = r-k
                                    pk = degree-k
                                    if r >= k:
                                        a3[s2,0] = a3[s1,0] * ndu3[pk+1,rk]
                                        d = a3[s2,0] * ndu3[rk,pk]
                                    j1 = 1   if (rk  > -1 ) else -rk
                                    j2 = k-1 if (r-1 <= pk) else degree-r
                                    for ij in range(j1,j2+1):
                                        a3[s2,ij] = (a3[s1,ij] - a3[s1,ij-1]) * ndu3[pk+1,rk+ij]
                                    for ij in range(j1,j2+1):
                                        d += a3[s2,ij]* ndu3[rk+ij,pk]
                                    if r <= pk:
                                        a3[s2,k] = - a3[s1,k-1] * ndu3[pk+1,r]
                                        d += a3[s2,k] * ndu3[r,pk]
                                    ders3[k,r] = d
                                    j  = s1
                                    s1 = s2
                                    s2 = j
                            # ...first compute R1
                            ders3[0,:]     = ders3[0,:] * omega_3[span-degree:]
                            sum_basisx    = sum(ders3[0,:])
                            basis_ad3[ie1, ie2, ie3, :, 0, g1, g2, g3] = ders3[0,:]/sum_basisx
                            r = degree
                            for i_ders in range(1,nders+1):
                                # Multiply derivatives by correct factors
                                ders3[i_ders,:] = ders3[i_ders,:] * r * omega_3[span-degree:]
                                basis_ad3[ie1, ie2, ie3, :, i_ders, g1, g2, g3] = ders3[i_ders,:]/sum_basisx
                                for j_ders in range(0,i_ders):
                                    basis_ad3[ie1, ie2, ie3, :, i_ders, g1, g2, g3] -= (j_ders+1)*basis_ad3[ie1, ie2, ie3, :, j_ders, g1, g2, g3] * sum(ders3[i_ders-j_ders,:])/sum_basisx
                                r = r * (degree-i_ders)