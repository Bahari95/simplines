__all__ = ['assemble_matrix_un_ex01',
           'assemble_vector_ex01',
           'assemble_norm_ex01'
]

from pyccel.decorators import types

#==============================================================================
# .. in uniform mesh Matrix
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        matrix):

    # ... sizes
    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1  = points_1[ie1,g1]
                    x2  = points_2[ie2,g2]

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v  = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]


                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]
                                    # ...
                                    v += (bj_x1* bi_x1 + bj_x2 * bi_x2) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1        = weights_1.shape[1]
    k2        = weights_2.shape[1]
    k3        = weights_3.shape[1]

    # ...
    lvalues_u = zeros((k1, k2, k3))

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            for ie3 in range(0, ne3):
                i_span_3 = spans_3[ie3]

                for il_1 in range(0, p1+1):
                  for il_2 in range(0, p2+1):
                    for il_3 in range(0, p3+1):

                      i1 = i_span_1 - p1 + il_1
                      i2 = i_span_2 - p2 + il_2
                      i3 = i_span_3 - p3 + il_3

                      v = 0.0
                      for g1 in range(0, k1):
                         for g2 in range(0, k2):
                            for g3 in range(0, k3):

                              bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                              #..
                              wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]
                              #..
                              t  = points_1[ie1, g1]
                              x  = points_2[ie2, g2]
                              y  = points_3[ie3, g3]
                              #.. Test 0
                              #u    = 3*pi**2*sin(pi*t)*sin(pi*x)*sin(pi*y)
                              #.. Test 0
                              u    =  51.0*pi**2*x**2*y*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) + 6*x**2*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) + 48.0*pi*x*y*(1.0 - y)*sin(pi*t)*cos(pi*(4.0 - 4.0*x)) - 6*y*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x))
                                                            
                              v   += bi_0 * u * wvol

                      rhs[i1+p1,i2+p2,i3+p3] += v   
    # ...

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:]', 'double[:,:,:]')
def assemble_norm_ex01(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, vector_u, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1         = weights_1.shape[1]
    k2         = weights_2.shape[1]
    k3         = weights_3.shape[1]

    # ...
    lcoeffs_u  = zeros((p1+1,p2+1,p3+1))
    # ...
    lvalues_u  = zeros((k1, k2, k3))
    lvalues_ut = zeros((k1, k2, k3))
    lvalues_ux = zeros((k1, k2, k3))
    lvalues_uy = zeros((k1, k2, k3))

    # ...
    norm_H1    = 0.
    norm_l2    = 0.
    for ie1 in range(0, ne1):
       i_span_1 = spans_1[ie1]
       for ie2 in range(0, ne2):
          i_span_2 = spans_2[ie2]
          for ie3 in range(0, ne3):
             i_span_3 = spans_3[ie3]

             lvalues_u[ : , : , :]  = 0.0
             lvalues_ut[ : , : , :] = 0.0
             lvalues_ux[ : , : , :] = 0.0
             lvalues_uy[ : , : , :] = 0.0
             lcoeffs_u[ : , : , :]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
             for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                   for il_3 in range(0, p3+1):
                      coeff_u = lcoeffs_u[il_1,il_2,il_3]

                      for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]
                            for g3 in range(0, k3):
                               b3  = basis_3[ie3,il_3,0,g3]
                               db3 = basis_3[ie3,il_3,1,g3]

                               lvalues_u[g1,g2,g3]  += coeff_u*b1*b2*b3
                               lvalues_ut[g1,g2,g3] += coeff_u*db1*b2*b3
                               lvalues_ux[g1,g2,g3] += coeff_u*b1*db2*b3
                               lvalues_uy[g1,g2,g3] += coeff_u*b1*b2*db3

             v = 0.0
             w = 0.0
             for g1 in range(0, k1):
               for g2 in range(0, k2):
                 for g3 in range(0, k3):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]

                    t  = points_1[ie1, g1]
                    x  = points_2[ie2, g2]
                    y  = points_3[ie3, g3]

                    # ... TEST 1 
                    #u    = sin(pi*t)*sin(pi*x)*sin(pi*y)
                    #ut   = pi*cos(pi*t)*sin(pi*x)*sin(pi*y)
                    #ux   = pi*sin(pi*t)*cos(pi*x)*sin(pi*y)
                    #uy   = pi*sin(pi*t)*sin(pi*x)*cos(pi*y)
                    # ... Test 2 
                    u    =  sin(pi*t)*x**2*y*3*sin(4.*pi*(1.-x))*(1.-y)
                    ut   = 3*pi*x**2*y*(1.0 - y)*sin(pi*(4.0 - 4.0*x))*cos(pi*t) 
                    ux   = -12.0*pi*x**2*y*(1.0 - y)*sin(pi*t)*cos(pi*(4.0 - 4.0*x)) + 6*x*y*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x))
                    uy   = -3*x**2*y*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) + 3*x**2*(1.0 - y)*sin(pi*t)*sin(pi*(4.0 - 4.0*x)) 
                    
                    uh  = lvalues_u[g1,g2,g3]
                    uht = lvalues_ut[g1,g2,g3]
                    uhx = lvalues_ux[g1,g2,g3]
                    uhy = lvalues_uy[g1,g2,g3]

                    v  += ((ut-uht)**2+(ux-uhx)**2+(uy-uhy)**2) * wvol
                    w  += (u-uh)**2 * wvol

             norm_H1 += v
             norm_l2 += w

    norm_H1 = sqrt(norm_H1)
    norm_l2 = sqrt(norm_l2)

    rhs[p1,p2,p3]   = norm_l2
    rhs[p1,p2,p3+1] = norm_H1
    # ...
