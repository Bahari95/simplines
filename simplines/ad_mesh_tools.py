from pyccel.decorators import types
from pyccel.epyccel    import epyccel
from .linalg           import StencilVector
from numpy             import zeros
from numpy             import double

#================================================================
#---2 : B-splines and thier corresponding spanes in adapted mesh 
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]','double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'float[:]', 'double[:,:]', 'double[:,:]', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]')
def assemble_basis_spans_in_adquadrature(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, knots_1, knots_2, knots_3, knots_4, knots_5, knots_6, vector_u, vector_w, spans_ad1, spans_ad2, basis_ad1, basis_ad2):

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

#==============================================================================
# ... in the same space L2 Bspline is the same as for Initial mapping
#==============================================================================
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
                 

class quadratur_in_admesh(object):
	'''
	The provided code calculates B-spline functions and their corresponding spans within the quadrature image using optimal mapping. 
	This mapping transforms a uniform mesh into an adaptive mesh within a unit square.
	'''
	def __init__(self, V) :
		# ...
		if V.dim == 6 :
			#.. separated spaces from optimal mapping
			self.basis_spans_in_adquadrature_2d = epyccel(assemble_basis_spans_in_adquadrature)
		else :
			# ... F lives in L2 degree-1 and optimal mapping lives in (degree, degree-1)x (degree-1, degree)
			self.basis_spans_in_adquadrature_2d = epyccel(assemble_basis_spans_in_adquadrature_same_space)
		# ...
		args  = []
		args += list(V.nelements)
		args += list(V.degree)
		args += list(V.spans)
		args += list(V.basis)
		args += list(V.weights)
		args += list(V.points)
		args += list(V.knots)
		# ...
		p1, p2       = V.degree[-2:]
		nx, ny       = V.nelements[-2:]
		wx, wy       = V.weights[-2:]
		# ...
		k1           = wx.shape[1]
		k2           = wy.shape[1]
		# ...
		nders        = 2
		self.args    = args
		self.nb_vec0 = (nx, ny, p1+1, nders+1, k1, k2)
		self.nb_vec1 = (nx, ny, p2+1, nders+1, k1, k2)
		self.ns_vec  = (nx, ny, k1, k2)
				
	def ad_quadrature(self, u01_mae, u10_mae):
		
		# ...
		if isinstance(u01_mae, StencilVector):
			# ...
			basis_ad1  = zeros(self.nb_vec0, dtype = double)
			basis_ad2  = zeros(self.nb_vec1, dtype = double)
			spans_ad1  = zeros(self.ns_vec, int)
			spans_ad2  = zeros(self.ns_vec, int)
			# ...
			self.basis_spans_in_adquadrature_2d(*self.args, u01_mae._data, u10_mae._data, spans_ad1, spans_ad2, basis_ad1, basis_ad2)
			return spans_ad1, spans_ad2, basis_ad1, basis_ad2
		else :
			print("Error please Try with StencilVector")
