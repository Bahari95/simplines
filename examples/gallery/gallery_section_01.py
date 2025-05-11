__all__ = ['assemble_matrix_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01',
           'assemble_matrix_ex02',
           'assemble_vector_ex02',
           'assemble_norm_ex02'
]

from pyccel.decorators import types

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]')
def assemble_vector_ex01(ne1, p1, spans_1, basis_1,  weights_1, points_1, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1        = weights_1.shape[1]
    # ...
    lvalues_u = zeros(k1)

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for il_1 in range(0, p1+1):
              i1 = i_span_1 - p1 + il_1

              v = 0.0
              for g1 in range(0, k1):

                      bi_0 = basis_1[ie1, il_1, 0, g1]
                      #..
                      wvol = weights_1[ie1, g1]
                      #..
                      x  = points_1[ie1, g1]
                      #.. Test 0
                      kappa= 1.
                      u    =  (pi)**2*sin(pi*x)
                                                    
                      v   += bi_0 * u * wvol

              rhs[i1+p1] += v   
    # ...

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:]', 'double[:]')
def assemble_norm_ex01(ne1, p1, spans_1, basis_1,  weights_1, points_1, vector_u, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1         = weights_1.shape[1]
    # ...
    lcoeffs_u  = zeros(p1+1)
    # ...
    lvalues_u  = zeros(k1)
    lvalues_ux = zeros(k1)

    # ...
    norm_H1    = 0.
    norm_l2    = 0.
    for ie1 in range(0, ne1):
       i_span_1 = spans_1[ie1]

       lvalues_u[ :]  = 0.0
       lvalues_ux[ :] = 0.0
       lcoeffs_u[ :]  = vector_u[i_span_1 : i_span_1+p1+1]
       for il_1 in range(0, p1+1):
              coeff_u = lcoeffs_u[il_1]

              for g1 in range(0, k1):
                 b1  = basis_1[ie1,il_1,0,g1]
                 db1 = basis_1[ie1,il_1,1,g1]
 
                 lvalues_u[g1]  += coeff_u*b1
                 lvalues_ux[g1] += coeff_u*db1

       v = 0.0
       w = 0.0
       for g1 in range(0, k1):
            wvol = weights_1[ie1, g1]

            x  = points_1[ie1, g1]

            # ... TEST 1
            kappa= 1. 
            u    = sin(pi*x)
            ux   = pi*cos(pi*x)
            # ... Test 2 
            
            uh  = lvalues_u[g1]
            uhx = lvalues_ux[g1]

            v  += (ux-uhx)**2 * wvol
            w  += (u-uh)**2 * wvol

       norm_H1 += v
       norm_l2 += w

    norm_H1 = sqrt(norm_H1)
    norm_l2 = sqrt(norm_l2)

    rhs[p1]   = norm_l2
    rhs[p1+1] = norm_H1
    # ...
