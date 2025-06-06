__all__ = ['assemble_matrix__un_ex01',
           'assemble_vector_un_ex01',
           'assemble_norm_ex02']

from pyccel.decorators import types

#... utilities of poisson equation

#==============================================================================
#---2 : In adapted mesh Matrix
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:,:]')
def assemble_matrix_un_ex01(ne1, ne2,
                        p1, p2,
                        spans_1, spans_2,
                        basis_1, basis_2,
                        weights_1, weights_2,
                        points_1, points_2,
                        vector_m1, vector_m2,
                        matrix):

    # ... sizes
    from numpy import zeros
    from numpy import sqrt
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #.. 

    # ...
    lcoeffs_m1 = zeros((p1+1,p2+1))
    lcoeffs_m2 = zeros((p1+1,p2+1))

    # ...
    arr_J_mat0 = zeros((k1,k2))
    arr_J_mat1 = zeros((k1,k2))
    arr_J_mat2 = zeros((k1,k2))
    arr_J_mat3 = zeros((k1,k2))
    J_mat      = zeros((k1,k2))

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_m1 = lcoeffs_m1[il_1,il_2]
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 = lcoeffs_m2[il_1,il_2]
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                    # ...
                    arr_J_mat0[g1,g2] = F2y
                    arr_J_mat1[g1,g2] = F1x
                    arr_J_mat2[g1,g2] = F1y
                    arr_J_mat3[g1,g2] = F2x

                    J_mat[g1,g2] = abs(F1x*F2y-F1y*F2x)

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):

                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1

                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2

                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]

                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bi_x  = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                                    bi_y  = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1

                                    bj_x  = arr_J_mat0[g1,g2] * bj_x1 - arr_J_mat3[g1,g2] * bj_x2 
                                    bj_y  = arr_J_mat1[g1,g2] * bj_x2 - arr_J_mat2[g1,g2] * bj_x1 


                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    += (bi_x * bj_x + bi_y * bj_y ) * wvol / J_mat[g1,g2]

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================
#---2 : rhs
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_un_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_m1, vector_m2, vector_d, rhs):

    from numpy import exp
    from numpy import cos, cosh
    from numpy import sin, sinh
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_m1  = zeros((p1+1,p2+1))
    lcoeffs_m2  = zeros((p1+1,p2+1))
    lcoeffs_di  = zeros((p1+1,p2+1))
    #..
    lvalues_u   = zeros((k1, k2))
    arr_J_mat0  = zeros((k1,k2))
    arr_J_mat1  = zeros((k1,k2))
    arr_J_mat2  = zeros((k1,k2))
    arr_J_mat3  = zeros((k1,k2))
    #lvalues_Jac = zeros((k1, k2))
    lvalues_udx = zeros((k1, k2))
    lvalues_udy = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_di[ : , : ] = vector_d[ i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x    = 0.0
                    y    = 0.0
                    F1x  = 0.0
                    F1y  = 0.0
                    F2x  = 0.0
                    F2y  = 0.0
                    # ...
                    udx  = 0.0
                    udy  = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_m1 =  lcoeffs_m1[il_1,il_2]
                              x       +=  coeff_m1 * bj_0
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 =  lcoeffs_m2[il_1,il_2]
                              y       +=  coeff_m2 * bj_0
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                              coeff_di =  lcoeffs_di[il_1,il_2]
                              udx     +=  coeff_di * bj_x
                              udy     +=  coeff_di * bj_y
                              
                    J_mat = abs(F1x*F2y-F1y*F2x)
                    arr_J_mat0[g1,g2] = F2y
                    arr_J_mat1[g1,g2] = F1x
                    arr_J_mat2[g1,g2] = F1y
                    arr_J_mat3[g1,g2] = F2x
                    lvalues_udx[g1, g2]  = (F2y * udx - F2x*udy)
                    lvalues_udx[g1, g2] /= J_mat
                    lvalues_udy[g1, g2]  = (F1x * udy - F1y*udx)
                    lvalues_udy[g1, g2] /= J_mat
                    #.. Test 1
                    f = 2.*(2.*pi)**2*sin(2.*pi*x)*sin(2.*pi*y)

                    lvalues_u[g1,g2]   = f * J_mat 
                    #lvalues_Jac[g1,g2] = J_mat
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            # ...
                            wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]
                            # ...
                            bi_x  = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                            bi_y  = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1
                            
                            u     = lvalues_u[g1,g2]
                            udx   = lvalues_udx[g1, g2]
                            udy   = lvalues_udy[g1, g2]
                            v    += bi_0 * u * wvol -  (udx * bi_x+ udy * bi_y) * wvol

                    rhs[i1+p1,i2+p2] += v


#=================================================================================
# norm in uniform mesh norm
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]',  'double[:,:]')
def assemble_norm_un_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_m1, vector_m2, vector_u, rhs):

    from numpy import exp
    from numpy import pi
    from numpy import sin, sinh
    from numpy import arctan2
    from numpy import cos, cosh
    from numpy import sqrt
    from numpy import zeros
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_m1 = zeros((p1+1,p2+1))
    lcoeffs_m2 = zeros((p1+1,p2+1))
    lcoeffs_u  = zeros((p1+1,p2+1))
    # ...
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
    lvalues_u  = zeros((k1, k2))

    error_l2 = 0.
    error_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ]   = 0.0
            lvalues_ux[ : , : ]  = 0.0
            lvalues_uy[ : , : ]  = 0.0
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]   += coeff_u*b1*b2
                            lvalues_ux[g1,g2]  += coeff_u*db1*b2
                            lvalues_uy[g1,g2]  += coeff_u*b1*db2

            w = 0.0
            v = 0.0
            lcoeffs_m1[ : , : ] = vector_m1[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_m2[ : , : ] = vector_m2[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x   = 0.0
                    y   = 0.0
                    F1x = 0.0
                    F1y = 0.0
                    F2x = 0.0
                    F2y = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_m1 =  lcoeffs_m1[il_1,il_2]
                              x       +=  coeff_m1 * bj_0
                              F1x     +=  coeff_m1 * bj_x
                              F1y     +=  coeff_m1 * bj_y

                              coeff_m2 =  lcoeffs_m2[il_1,il_2]
                              y       +=  coeff_m2 * bj_0
                              F2x     +=  coeff_m2 * bj_x
                              F2y     +=  coeff_m2 * bj_y

                    det_J = abs(F1x*F2y-F1y*F2x)
                    
                    # ...                              
                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2] * det_J

                    uh    = lvalues_u[g1,g2]
                    sx    = lvalues_ux[g1,g2]
                    sy    = lvalues_uy[g1,g2]
 
                    #... 
                    f    = sin(2.*pi*x)*sin(2.*pi*y)
                    fx   = 2.*pi*cos(2.*pi*x)*sin(2.*pi*y)
                    fy   = 2.*pi*sin(2.*pi*x)*cos(2.*pi*y)
                    # ...
                    uhx   = (F2y*sx-F2x*sy)/det_J
                    uhy   = (F1x*sy-F1y*sx)/det_J

                    w    += ((uhx-fx)**2 +(uhy-fy)**2)* wvol
                    v    += (uh-f)**2 * wvol

            error_H1      += w
            error_l2      += v
    rhs[p1,p2]   = sqrt(error_l2)
    rhs[p1,p2+1] = sqrt(error_H1)
    #...