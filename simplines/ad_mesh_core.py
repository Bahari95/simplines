from   pyccel.decorators import types

#=====================================================================================
# ... The Hdiv mapping space can be selected independently of the initial mapping space.
#=====================================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'double[:,:]', 'double[:,:]', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]')
def assemble_basis_spans_in_adquadrature(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, knots_1, knots_2, knots_3, knots_4, knots_5, knots_6, vector_u, vector_w, spans_ad1, spans_ad2, basis_ad1, basis_ad2):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    from numpy import empty
    # ...
    k1 = weights_5.shape[1]
    k2 = weights_6.shape[1]

    # ___
    lcoeffs_u   = zeros((p1+1,p3+1))
    lcoeffs_w   = zeros((p4+1,p2+1))

    # ... Initialization
    points1    = zeros((ne1, ne2, k1, k2))
    points2    = zeros((ne1, ne2, k1, k2))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p3+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,0,g2]
                              coeff_u = lcoeffs_u[il_1,il_2]

                              sx     +=  coeff_u*bj_0
                    sy = 0.0
                    for il_1 in range(0, p4+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              coeff_w = lcoeffs_w[il_1,il_2]

                              sy     += coeff_w*bj_0
                    points1[ie1, ie2, g1, g2] = sx
                    points2[ie1, ie2, g1, g2] = sy

    #   ---Computes All basis in a new points
    nders          = 2
    degree         = p5
    # ...
    left           = empty( degree )
    right          = empty( degree )
    a              = empty( (       2, degree+1) )
    ndu            = empty( (degree+1, degree+1) )
    ders           = zeros( (     nders+1, degree+1) ) # output array
    #basis5         = zeros( (ne, degree+1, nders+1, nq))
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points1[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left/right boundary
                 low  = degree
                 high = len(knots_5)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
                 if xq <= knots_5[low ]: 
                      span = low
                 elif xq >= knots_5[high]: 
                      span = high-1
                 else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_5[span] or xq >= knots_5[span+1]:
                      if xq < knots_5[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                 # ... 
                 spans_ad1[ie1, ie2, g1, g2] = span
                 # ...
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_5[span-j]
                     right[j] = knots_5[span+1+j] - xq
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad1[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad1[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad1[ie1, ie2, :, 2, g1, g2] = ders[2,:]

    degree         = p6
    #...
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points2[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left/right boundary
                 low  = degree
                 high = len(knots_6)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
                 if xq <= knots_6[low ]: 
                      span = low
                 elif xq >= knots_6[high]: 
                      span = high-1
                 else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_6[span] or xq >= knots_6[span+1]:
                      if xq < knots_6[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                 # ... 
                 spans_ad2[ie1, ie2, g1, g2] = span
                 # ...
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_6[span-j]
                     right[j] = knots_6[span+1+j] - xq
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad2[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad2[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad2[ie1, ie2, :, 2, g1, g2] = ders[2,:]

#===========================================================================
# ... L2-B-spline space for Hdiv mapping is the same as for initial mapping.
#===========================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
@types( 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'double[:,:]', 'double[:,:]', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]')
def assemble_basis_spans_in_adquadrature_same_space(ne1, ne2, ne3, ne4, p1, p2, p3, p4, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, weights_1, weights_2, weights_3, weights_4, points_1, points_2, points_3, points_4, knots_1, knots_2, knots_3, knots_4, vector_u, vector_w, spans_ad1, spans_ad2, basis_ad1, basis_ad2):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    from numpy import empty
    # ...
    k1 = weights_3.shape[1]
    k2 = weights_4.shape[1]

    # ___
    lcoeffs_u   = zeros((p1+1,p3+1))
    lcoeffs_w   = zeros((p4+1,p2+1))

    # ... Initialization
    points1    = zeros((ne1, ne2, k1, k2))
    points2    = zeros((ne1, ne2, k1, k2))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p3+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_3[ie2,il_2,0,g2]
                              coeff_u = lcoeffs_u[il_1,il_2]

                              sx     +=  coeff_u*bj_0
                    sy = 0.0
                    for il_1 in range(0, p4+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_4[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              coeff_w = lcoeffs_w[il_1,il_2]

                              sy     += coeff_w*bj_0
                    points1[ie1, ie2, g1, g2] = sx
                    points2[ie1, ie2, g1, g2] = sy

    #   ---Computes All basis in a new points
    nders          = 2
    degree         = p3
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
                 high = len(knots_3)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
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
                 spans_ad1[ie1, ie2, g1, g2] = span
                 # ...
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_3[span-j]
                     right[j] = knots_3[span+1+j] - xq
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad1[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad1[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad1[ie1, ie2, :, 2, g1, g2] = ders[2,:]

    degree         = p4
    #...
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points2[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left/right boundary
                 low  = degree
                 high = len(knots_4)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
                 if xq <= knots_4[low ]: 
                      span = low
                 elif xq >= knots_4[high]: 
                      span = high-1
                 else : 
                   # Perform binary search
                   span = (low+high)//2
                   while xq < knots_4[span] or xq >= knots_4[span+1]:
                      if xq < knots_4[span]:
                          high = span
                      else:
                          low  = span
                      span = (low+high)//2
                 # ... 
                 spans_ad2[ie1, ie2, g1, g2] = span
                 # ...
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_4[span-j]
                     right[j] = knots_4[span+1+j] - xq
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad2[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad2[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad2[ie1, ie2, :, 2, g1, g2] = ders[2,:]
                 
#==============================================================================
# ... L2-B-spline space for Gradien mapping is the same as for initial mapping.
#==============================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'double[:,:]', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]')
def assemble_basis_spans_in_adquadrature_gradmap(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, knots_1, knots_2, vector_u, spans_ad1, spans_ad2, basis_ad1, basis_ad2):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    from numpy import empty
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ___
    lcoeffs_u   = zeros((p1+1,p2+1))

    # ... Initialization
    points1    = zeros((ne1, ne2, k1, k2))
    points2    = zeros((ne1, ne2, k1, k2))

    # ... Assemble a new points by a new map
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x    = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_u = lcoeffs_u[il_1,il_2]

                              sx     += coeff_u * bj_x
                              sy     += coeff_u * bj_y
                              
                    points1[ie1, ie2, g1, g2] = sx
                    points2[ie1, ie2, g1, g2] = sy

    #   ---Computes All basis in a new points
    nders          = 2
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad1[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad1[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad1[ie1, ie2, :, 2, g1, g2] = ders[2,:]

    degree         = p2
    #...
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points2[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left/right boundary
                 low  = degree
                 high = len(knots_2)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
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
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_2[span-j]
                     right[j] = knots_2[span+1+j] - xq
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad2[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad2[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad2[ie1, ie2, :, 2, g1, g2] = ders[2,:]
#==============================================================================
# ... L2-B-spline space for L^2mapping is the same as for initial mapping.
#==============================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'double[:,:]', 'double[:,:]', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]')
def assemble_basis_spans_in_adquadrature_L2map(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, knots_1, knots_2, vector_u1, vector_u2, spans_ad1, spans_ad2, basis_ad1, basis_ad2):

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
    nders          = 2
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad1[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad1[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad1[ie1, ie2, :, 2, g1, g2] = ders[2,:]

    degree         = p2
    #...
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
          for g1 in range(0, k1):
             for g2 in range(0, k2):
                 xq = points2[ie1, ie2, g1, g2]

                 #span = find_span( knots, degree, xq )
                 #~~~~~~~~~~~~~~~
                 # Knot index at left/right boundary
                 low  = degree
                 high = len(knots_2)-1-degree
                 # Check if point is exactly on left/right boundary, or outside domain
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
                 ndu[0,0] = 1.0
                 for j in range(0,degree):
                     left [j] = xq - knots_2[span-j]
                     right[j] = knots_2[span+1+j] - xq
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
                 # Multiply derivatives by correct factors
                 r = degree
                 ders[1,:] = ders[1,:] * r
                 r = r * (degree-1)
                 ders[2,:] = ders[2,:] * r
                 # ...
                 basis_ad2[ie1, ie2, :, 0, g1, g2] = ders[0,:]
                 basis_ad2[ie1, ie2, :, 1, g1, g2] = ders[1,:]
                 basis_ad2[ie1, ie2, :, 2, g1, g2] = ders[2,:]

#==============================================================================
# ... L2-B-spline space for L^2mapping is the same as for initial mapping.
#==============================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'float[:]', 'double[:,:,:]', 'double[:,:,:]', 'double[:,:,:]', 'int[:,:,:,:,:,:]', 'int[:,:,:,:,:,:]', 'int[:,:,:,:,:,:]', 'double[:,:,:,:,:,:,:,:]', 'double[:,:,:,:,:,:,:,:]', 'double[:,:,:,:,:,:,:,:]', 'int')
def assemble_basis_spans_in_adquadrature_3L2map(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, knots_1, knots_2, knots_3, vector_u1, vector_u2, vector_u3, spans_ad1, spans_ad2, spans_ad3, basis_ad1, basis_ad2, basis_ad3, nders):

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
                            # Multiply derivatives by correct factors
                            r = degree
                            ders[1,:] = ders[1,:] * r
                            r = r * (degree-1)
                            ders[2,:] = ders[2,:] * r
                            # ...
                            for i_ders in range(nders):
                                basis_ad1[ie1, ie2, ie3, :, i_ders, g1, g2, g3] = ders[i_ders,:]

    degree         = p2
    #...
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
           for ie3 in range(0, ne3):
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        for g3 in range(0, k3):
                            xq = points2[ie1, ie2, ie3, g1, g2, g3]

                            #span = find_span( knots, degree, xq )
                            #~~~~~~~~~~~~~~~
                            # Knot index at left/right boundary
                            low  = degree
                            high = len(knots_2)-1-degree
                            # Check if point is exactly on left/right boundary, or outside domain
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
                            ndu[0,0] = 1.0
                            for j in range(0,degree):
                                left [j] = xq - knots_2[span-j]
                                right[j] = knots_2[span+1+j] - xq
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
                            # Multiply derivatives by correct factors
                            r = degree
                            ders[1,:] = ders[1,:] * r
                            r = r * (degree-1)
                            ders[2,:] = ders[2,:] * r
                            # ...
                            for i_ders in range(nders):
                                basis_ad2[ie1, ie2, ie3, :, i_ders, g1, g2, g3] = ders[i_ders,:]
    degree         = p3
    #...
    for ie1 in range(0, ne1):
       for ie2 in range(0, ne2):
            for ie3 in range(0, ne3):
                for g1 in range(0, k1):
                    for g2 in range(0, k2):
                        for g3 in range(0, k3):
                            xq = points2[ie1, ie2, ie3, g1, g2, g3]

                            #span = find_span( knots, degree, xq )
                            #~~~~~~~~~~~~~~~
                            # Knot index at left/right boundary
                            low  = degree
                            high = len(knots_3)-1-degree
                            # Check if point is exactly on left/right boundary, or outside domain
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
                            ndu[0,0] = 1.0
                            for j in range(0,degree):
                                left [j] = xq - knots_3[span-j]
                                right[j] = knots_3[span+1+j] - xq
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
                            # Multiply derivatives by correct factors
                            r = degree
                            ders[1,:] = ders[1,:] * r
                            r = r * (degree-1)
                            ders[2,:] = ders[2,:] * r
                            # ...
                            for i_ders in range(nders):
                                basis_ad3[ie1, ie2, ie3, :, i_ders, g1, g2, g3] = ders[i_ders,:]

# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v   += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v

# assembles mass matrix 1D
#=============================================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points, matrix):

    # ... sizes
    k1 = weights.shape[1]
    # ... build matrices
    for ie1 in range(0, ne):
            i_span_1 = spans[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, degree+1):
                i1 = i_span_1 - degree + il_1
                for il_2 in range(0, degree+1):
                            i2 = i_span_1 - degree + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v   += bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v
    # ...

@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex01(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]  
            i_span_2 = spans_2[ie1]      
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):
                            i2 = i_span_2 - p2 + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis_1[ie1, il_1, 1, g1]
                                    bj_0 = basis_2[ie1, il_2, 0, g1]
                                    
                                    wvol = weights_1[ie1, g1]
                                    
                                    v   += bi_x * bj_0 * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...


@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_matrix_ex02(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]

    # ... build matrices
    for ie1 in range(0, ne1):
            i_span_1 = spans_1[ie1]
            i_span_2 = spans_2[ie1]        
            # evaluation dependant uniquement de l'element

            for il_1 in range(0, p1+1):
                i1 = i_span_1 - p1 + il_1
                for il_2 in range(0, p2+1):
                            i2 = i_span_2 - p2 + il_2
                            v  = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis_1[ie1, il_1, 0, g1]
                                    bj_x = basis_2[ie1, il_2, 1, g1]
                                    
                                    wvol = weights_1[ie1, g1]
                                    
                                    v   += bi_0 * bj_x * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...