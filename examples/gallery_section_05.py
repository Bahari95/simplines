__all__ = ['assemble_matrix_ex11',
           'assemble_matrix_ex12',
           'assemble_vector_ex22',
           'assemble_vector_ex21',
           'assemble_norm_ex02']

from pyccel.decorators import types

#... utilities of poisson equation
#==============================================================================
#---2 : A11
@types( 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'double[:,:,:,:]')
def assemble_matrix_ad_ex11(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2, weights_1, weights_2,  points_1, points_2, vector_u, vector_w, mu, lanbda, matrix):

    # ... sizes
    from numpy import sqrt
    from numpy import zeros
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    lcoeffs_w  = zeros((p1+1,p2+1))
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

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sxx = 0.0
                    sxy = 0.0
                    syy = 0.0
                    syx = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x    = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_u = lcoeffs_u[il_1,il_2]

                              sxx     +=  coeff_u*bj_x
                              sxy     +=  coeff_u*bj_y

                              coeff_w = lcoeffs_w[il_1,il_2]

                              syy    +=  coeff_w*bj_y
                              syx    +=  coeff_w*bj_x
                              
                    arr_J_mat0[g1,g2] = syy
                    arr_J_mat1[g1,g2] = sxx
                    arr_J_mat2[g1,g2] = sxy
                    arr_J_mat3[g1,g2] = syx
                    J_mat[g1,g2]      = abs(sxx*syy-sxy*syx)

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
                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bi_x = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                                    bi_y = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1

                                    bj_x = arr_J_mat0[g1,g2] * bj_x1 - arr_J_mat3[g1,g2] * bj_x2 
                                    bj_y = arr_J_mat1[g1,g2] * bj_x2 - arr_J_mat2[g1,g2] * bj_x1 

                                    # ...
                                    wvol = weights_2[ie1, g1] * weights_2[ie2, g2] / J_mat[g1,g2]
                                    # ...mu, lanbda
                                    v += (lanbda * bj_x * bi_x + mu * bj_y * bi_y) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...A22  v += (mu * bj_x * bi_x + lanbda * bj_y * bi_y) * wvol

#==============================================================================
#---2 : A12
@types( 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'double[:,:,:,:]')
def assemble_matrix_ad_ex12(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2, weights_1, weights_2,  points_1, points_2, vector_u, vector_w, mu, lanbda, matrix):

    # ... sizes
    from numpy import sqrt
    from numpy import zeros
    # ...
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    lcoeffs_w  = zeros((p1+1,p2+1))
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

            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    sxx = 0.0
                    sxy = 0.0
                    syy = 0.0
                    syx = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_x    = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_u = lcoeffs_u[il_1,il_2]

                              sxx     +=  coeff_u*bj_x
                              sxy     +=  coeff_u*bj_y

                              coeff_w = lcoeffs_w[il_1,il_2]

                              syy    +=  coeff_w*bj_y
                              syx    +=  coeff_w*bj_x
                              
                    arr_J_mat0[g1,g2] = syy
                    arr_J_mat1[g1,g2] = sxx
                    arr_J_mat2[g1,g2] = sxy
                    arr_J_mat3[g1,g2] = syx
                    J_mat[g1,g2]      = abs(sxx*syy-sxy*syx)

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
                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bi_x = arr_J_mat0[g1,g2] * bi_x1 - arr_J_mat3[g1,g2] * bi_x2
                                    bi_y = arr_J_mat1[g1,g2] * bi_x2 - arr_J_mat2[g1,g2] * bi_x1

                                    bj_x = arr_J_mat0[g1,g2] * bj_x1 - arr_J_mat3[g1,g2] * bj_x2 
                                    bj_y = arr_J_mat1[g1,g2] * bj_x2 - arr_J_mat2[g1,g2] * bj_x1 

                                    # ...
                                    wvol = weights_2[ie1, g1] * weights_2[ie2, g2] / J_mat[g1,g2]
                                    # ...mu, lanbda
                                    v += (mu * bj_x * bi_y + lanbda * bj_y * bi_x) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ... A21 v += (mu * bj_y * bi_x + lanbda * bj_x * bi_y ) * wvol

#==============================================================================
#---2 : rhs1
@types( 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'double[:,:]')
def assemble_vector_ex12(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, vector_u, vector_w, Tx, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u   = zeros((p1+1,p2+1))
    lcoeffs_w   = zeros((p1+1,p2+1))
    lcoeffs_d   = zeros((p1+1,p2+1))
    # ..
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        i_span_2 = spans_2[ie2]
           
        ie1      = 0
        i_span_1 = spans_1[ie1]
        
        lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0  = 0.0
           for g2 in range(0, k2):

                  # ...
                  x1  = 0.0
                  x2  = 0.0
                  sxy = 0.0
                  syy = 0.0
                  for jl_2 in range(0, p2+1):

                              bj_0    =  basis_2[ie2,jl_2,0,g2]
                              bj_y    =  basis_2[ie2,jl_2,1,g2]

                              coeff_u = lcoeffs_u[0,jl_2]

                              x1     +=  coeff_u*bj_0
                              sxy    +=  coeff_u*bj_y

                              coeff_w = lcoeffs_w[0,jl_2]
                              
                              x2     +=  coeff_w*bj_0
                              syy    +=  coeff_w*bj_y
                              
                  bi_0     =  basis_2[ie2, il_2, 0, g2]
                  # ... Where \xi=1
                  R_2      = 1.0
                  r_q      = x1**2+x2**2 
                  r_f      = R_2/r_q
                  
                  sigma_rr = 0.5*Tx*(1.-R_2/r_q + (1.-4.*R_2/r_q+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_oo = 0.5*Tx*(1.+R_2/r_q - (1.+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_ro = -1*0.5*Tx*(1.+2.*R_2/r_q -3.*R_2**2/r_q**2)*(2.*x1*x2)/r_q
                  # ...
                  sigma_xy = (x1*x2/r_q)*sigma_rr - (x1*x2/r_q)*sigma_oo + ((x1**2-x2**2)/r_q)*sigma_ro 
                  sigma_yy = (x2**2/r_q)*sigma_rr + (x1**2/r_q)*sigma_oo + 2.*(x1*x2/r_q)*sigma_ro 
                  sigma_xx = (x1**2/r_q)*sigma_rr + (x2**2/r_q)*sigma_oo - 2.*(x1*x2/r_q)*sigma_ro 
                  #cos_2o    = (x1**2-x2**2)/r_q
                  #sin_2o    = 2.*x1*x2/r_q
                  #cos_4o    = cos_2o**2 - sin_2o**2
                  #sin_4o    = 2.*cos_2o*sin_2o
                  
                  #sigma_xx = 1.- r_f * (3./2.*cos_2o + cos_4o) + 3./2.*r_f**2 * cos_4o
                  #sigma_yy = -1.*r_f * (1./2.*cos_2o - cos_4o) - 3./2.*r_f**2 * cos_4o
                  #sigma_xy = -1.*r_f * (1./2.*sin_2o + sin_4o) + 3./2.*r_f**2 * sin_4o
                  
                  P_sc1    = -syy*sigma_xx + sxy*sigma_xy
                  #P_sc1    = (x1*sigma_xx + x2*sigma_xy)
                  # ...
                  wleng_y  = weights_2[ie2, g2] #* sqrt(sxy**2 + syy**2)
                  # ...
                  vy_0    += P_sc1 * bi_0 * wleng_y
           rhs[p1,i2+p2]  += 0.

        ie1      = ne1 - 1
        i_span_1 = spans_1[ie1]
        
        lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_1  = 0.0
           for g2 in range(0, k2):

                  # ...
                  x1  = 0.0
                  x2  = 0.0
                  sxy = 0.0
                  syy = 0.0
                  for jl_2 in range(0, p2+1):

                              bj_0    =  basis_2[ie2,jl_2,0,g2]
                              bj_y    =  basis_2[ie2,jl_2,1,g2]

                              coeff_u = lcoeffs_u[p1,jl_2]

                              x1     +=  coeff_u*bj_0
                              sxy    +=  coeff_u*bj_y

                              coeff_w = lcoeffs_w[p1,jl_2]
                              
                              x2     +=  coeff_w*bj_0
                              syy    +=  coeff_w*bj_y
                              
                  bi_0     =  basis_2[ie2, il_2, 0, g2]
                  # ... Where \xi=1
                  R_2      = 1.0
                  r_q      = x1**2+x2**2 
                  r_f      = R_2/r_q
                  
                  sigma_rr = 0.5*Tx*(1.-R_2/r_q + (1.-4.*R_2/r_q+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_oo = 0.5*Tx*(1.+R_2/r_q - (1.+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_ro = -1*0.5*Tx*(1.+2.*R_2/r_q -3.*R_2**2/r_q**2)*(2.*x1*x2)/r_q
                  # ...
                  sigma_xy = (x1*x2/r_q)*sigma_rr - (x1*x2/r_q)*sigma_oo + ((x1**2-x2**2)/r_q)*sigma_ro 
                  sigma_yy = (x2**2/r_q)*sigma_rr + (x1**2/r_q)*sigma_oo + 2.*(x1*x2/r_q)*sigma_ro 
                  sigma_xx = (x1**2/r_q)*sigma_rr + (x2**2/r_q)*sigma_oo - 2.*(x1*x2/r_q)*sigma_ro 
                  #cos_2o    = (x1**2-x2**2)/r_q
                  #sin_2o    = 2.*x1*x2/r_q
                  #cos_4o    = cos_2o**2 - sin_2o**2
                  #sin_4o    = 2.*cos_2o*sin_2o
                  
                  #sigma_xx = 1.- r_f * (3./2.*cos_2o + cos_4o) + 3./2.*r_f**2 * cos_4o
                  #sigma_yy = -1.*r_f * (1./2.*cos_2o - cos_4o) - 3./2.*r_f**2 * cos_4o
                  #sigma_xy = -1.*r_f * (1./2.*sin_2o + sin_4o) + 3./2.*r_f**2 * sin_4o
                  
                  P_sc1    = syy*sigma_xx - sxy*sigma_xy
                  #P_sc1    = (x1*sigma_xx + x2*sigma_xy)
                  # ...
                  wleng_y  = weights_2[ie2, g2] #* sqrt(sxy**2 + syy**2)
                  # ...
                  vy_1    += P_sc1 * bi_0 * wleng_y
           rhs[ne1-1+2*p1,i2+p2] += vy_1
    # ...

#==============================================================================
#---2 : rhs2
@types( 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'double[:,:]')
def assemble_vector_ex22(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, vector_u, vector_w, Tx, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]

    # ...
    lcoeffs_u   = zeros((p1+1,p1+1))
    lcoeffs_w   = zeros((p1+1,p2+1))
    lcoeffs_d   = zeros((p1+1,p2+1))
    # ..
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        i_span_2 = spans_2[ie2]
           
        ie1      = 0
        i_span_1 = spans_1[ie1]
        
        lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0  = 0.0
           for g2 in range(0, k2):

                  # ...
                  x1  = 0.0
                  x2  = 0.0
                  sxy = 0.0
                  syy = 0.0
                  for jl_2 in range(0, p2+1):

                              bj_0    =  basis_2[ie2,jl_2,0,g2]
                              bj_y    =  basis_2[ie2,jl_2,1,g2]

                              coeff_u = lcoeffs_u[0,jl_2]

                              x1     +=  coeff_u*bj_0
                              sxy    +=  coeff_u*bj_y

                              coeff_w = lcoeffs_w[0,jl_2]
                              
                              x2     +=  coeff_w*bj_0
                              syy    +=  coeff_w*bj_y
                              
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  # ... Where \xi=1
                  R_2      = 1.0
                  r_q      = x1**2+x2**2 
                  r_f      = R_2/r_q
                  
                  sigma_rr = 0.5*Tx*(1.-R_2/r_q + (1.-4.*R_2/r_q+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_oo = 0.5*Tx*(1.+R_2/r_q - (1.+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_ro = -1*0.5*Tx*(1.+2.*R_2/r_q -3.*R_2**2/r_q**2)*(2.*x1*x2)/r_q
                  # ...
                  sigma_xy = (x1*x2/r_q)*sigma_rr - (x1*x2/r_q)*sigma_oo + ((x1**2-x2**2)/r_q)*sigma_ro 
                  sigma_yy = (x2**2/r_q)*sigma_rr + (x1**2/r_q)*sigma_oo + 2.*(x1*x2/r_q)*sigma_ro 
                  sigma_xx = (x1**2/r_q)*sigma_rr + (x2**2/r_q)*sigma_oo - 2.*(x1*x2/r_q)*sigma_ro 
                  #cos_2o    = (x1**2-x2**2)/r_q
                  #sin_2o    = 2.*x1*x2/r_q
                  #cos_4o    = cos_2o**2 - sin_2o**2
                  #sin_4o    = 2.*cos_2o*sin_2o
                  
                  #sigma_xx = 1.- r_f * (3./2.*cos_2o + cos_4o) + 3./2.*r_f**2 * cos_4o
                  #sigma_yy = -1.*r_f * (1./2.*cos_2o - cos_4o) - 3./2.*r_f**2 * cos_4o
                  #sigma_xy = -1.*r_f * (1./2.*sin_2o + sin_4o) + 3./2.*r_f**2 * sin_4o
                                    
                  P_sc2   =  - syy*sigma_xy + sxy*sigma_yy
                  #P_sc2   =  (x1*sigma_xy + x2*sigma_yy)
                  # ...
                  wleng_y = weights_2[ie2, g2] #* sqrt(sxy**2 + syy**2)
                  # ...
                  vy_0   +=  P_sc2 * bi_0 * wleng_y
           rhs[p1,i2+p2] += 0.

        ie1      = ne1 - 1
        i_span_1 = spans_1[ie1]
        
        lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_1  = 0.0
           for g2 in range(0, k2):

                  # ...
                  x1  = 0.0
                  x2  = 0.0
                  sxy = 0.0
                  syy = 0.0
                  for jl_2 in range(0, p2+1):

                              bj_0    =  basis_2[ie2,jl_2,0,g2]
                              bj_y    =  basis_2[ie2,jl_2,1,g2]

                              coeff_u = lcoeffs_u[p1,jl_2]

                              x1     +=  coeff_u*bj_0
                              sxy    +=  coeff_u*bj_y

                              coeff_w = lcoeffs_w[p1,jl_2]
                              
                              x2     +=  coeff_w*bj_0
                              syy    +=  coeff_w*bj_y
                              
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  # ... Where \xi=1
                  R_2      = 1.0
                  r_q      = x1**2+x2**2 
                  r_f      = R_2/r_q
                  
                  sigma_rr = 0.5*Tx*(1.-R_2/r_q + (1.-4.*R_2/r_q+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_oo = 0.5*Tx*(1.+R_2/r_q - (1.+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                  sigma_ro = -1*0.5*Tx*(1.+2.*R_2/r_q -3.*R_2**2/r_q**2)*(2.*x1*x2)/r_q
                  # ...
                  sigma_xy = (x1*x2/r_q)*sigma_rr - (x1*x2/r_q)*sigma_oo + ((x1**2-x2**2)/r_q)*sigma_ro 
                  sigma_yy = (x2**2/r_q)*sigma_rr + (x1**2/r_q)*sigma_oo + 2.*(x1*x2/r_q)*sigma_ro 
                  sigma_xx = (x1**2/r_q)*sigma_rr + (x2**2/r_q)*sigma_oo - 2.*(x1*x2/r_q)*sigma_ro 
                  #cos_2o    = (x1**2-x2**2)/r_q
                  #sin_2o    = 2.*x1*x2/r_q
                  #cos_4o    = cos_2o**2 - sin_2o**2
                  #sin_4o    = 2.*cos_2o*sin_2o
                  
                  #sigma_xx = 1.- r_f * (3./2.*cos_2o + cos_4o) + 3./2.*r_f**2 * cos_4o
                  #sigma_yy = -1.*r_f * (1./2.*cos_2o - cos_4o) - 3./2.*r_f**2 * cos_4o
                  #sigma_xy = -1.*r_f * (1./2.*sin_2o + sin_4o) + 3./2.*r_f**2 * sin_4o
                                    
                  P_sc2   =   syy*sigma_xy - sxy*sigma_yy
                  #P_sc2   =  (x1*sigma_xy + x2*sigma_yy)
                  # ...
                  wleng_y = weights_2[ie2, g2] #* sqrt(sxy**2 + syy**2)
                  # ...
                  vy_1   +=  P_sc2 * bi_0 * wleng_y
           rhs[ne1-1+2*p1,i2+p2] += vy_1
    # ...
#==============================================================================
#---2 : In adapted mesh
@types( 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'double[:,:]')
def assemble_norm_ex02(ne1, ne2, p1, p2, spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, vector_u, vector_w, vector_1z, vector_2z,  mu, lanbda, Tx, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros
    
    # ... sizes
    k1           = weights_1.shape[1]
    k2           = weights_2.shape[1]
    # ...
    lcoeffs_u    = zeros((p1+1,p2+1))
    lcoeffs_w    = zeros((p1+1,p2+1))
    lcoeffs_1z   = zeros((p1+1,p2+1))
    lcoeffs_2z   = zeros((p1+1,p2+1))
        
    lvalues_1u   = zeros( k2)
    lvalues_1ux  = zeros( k2)
    lvalues_1uy  = zeros( k2)
    # ...
    lvalues_2u   = zeros( k2)
    lvalues_2ux  = zeros( k2)
    lvalues_2uy  = zeros( k2)
    # ...
    lvalues_u1   = zeros((k1, k2))
    lvalues_u1x  = zeros((k1, k2))
    lvalues_u1y  = zeros((k1, k2))
    lvalues_u2   = zeros((k1, k2))
    lvalues_u2x  = zeros((k1, k2))
    lvalues_u2y  = zeros((k1, k2))
    
    lvalues_1sx  = zeros((k1, k2))
    lvalues_1sy  = zeros((k1, k2))
    lvalues_2sx  = zeros((k1, k2))
    lvalues_2sy  = zeros((k1, k2))
    # ...
    norm_l2      = 0.                                
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_1 = spans_1[ie1]
        
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_2 = spans_2[ie2]

            lvalues_u1[ : , : ]  = 0.0
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0

            lvalues_u2[ : , : ]  = 0.0
            lvalues_u2x[ : , : ] = 0.0
            lvalues_u2y[ : , : ] = 0.0
            
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_w[ : , : ]   = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]
                    coeff_w = lcoeffs_w[il_1,il_2]
                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_2[ie2,il_2,0,g2] 
                            db2  = basis_2[ie2,il_2,1,g2] 

                            lvalues_u1[g1,g2]  += coeff_u* b1*b2
                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u* b1*db2

                            lvalues_u2[g1,g2]  += coeff_w*b1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2
                            lvalues_u2x[g1,g2] += coeff_w*db1*b2

            lvalues_1sx[ : , : ]     = 0.0
            lvalues_1sy[ : , : ]     = 0.0

            lvalues_2sx[ : , : ]     = 0.0
            lvalues_2sy[ : , : ]     = 0.0

            lcoeffs_1z[ : , : ] = vector_1z[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_2z[ : , : ] = vector_2z[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_1z = lcoeffs_1z[il_1,il_2]
                    coeff_2z = lcoeffs_2z[il_1,il_2]
                    
                    for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_2[ie2,il_2,0,g2]
                            db2  = basis_2[ie2,il_2,1,g2]

                            lvalues_1sx[g1,g2] += coeff_1z*db1*b2
                            lvalues_1sy[g1,g2] += coeff_1z*b1*db2
                            lvalues_2sx[g1,g2] += coeff_2z*db1*b2
                            lvalues_2sy[g1,g2] += coeff_2z*b1*db2

            v = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x1       = lvalues_u1[g1,g2]
                    x2       = lvalues_u2[g1,g2]
                    # ...
                    R_2      = 1.0
                    r_q      = x1**2+x2**2 
                    r_f      = R_2/r_q
                    sigma_rr = 0.5*Tx*(1.-R_2/r_q + (1.-4.*R_2/r_q+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                    sigma_oo = 0.5*Tx*(1.+R_2/r_q - (1.+3.*R_2**2/r_q**2)*(x1**2-x2**2)/r_q)
                    sigma_ro = -1*0.5*Tx*(1.+2.*R_2/r_q -3.*R_2**2/r_q**2)*(2.*x1*x2)/r_q
                    # ...
                    sigma_yy = (x2**2/r_q)*sigma_rr + (x1**2/r_q)*sigma_oo + 2.*(x1*x2/r_q)*sigma_ro 
                    sigma_xx = (x1**2/r_q)*sigma_rr + (x2**2/r_q)*sigma_oo - 2.*(x1*x2/r_q)*sigma_ro 
                    sigma_xy = (x1*x2/r_q)*sigma_rr - (x1*x2/r_q)*sigma_oo + ((x1**2-x2**2)/r_q)*sigma_ro 
                    
                    #cos_2o    = (x1**2-x2**2)/r_q
                    #sin_2o    = 2.*x1*x2/r_q
                    #cos_4o    = cos_2o**2 - sin_2o**2
                    #sin_4o    = 2.*cos_2o*sin_2o
                  
                    #sigma_xx = 1.- r_f * (3./2.*cos_2o + cos_4o) + 3./2.*r_f**2 * cos_4o
                    #sigma_yy = -1.*r_f * (1./2.*cos_2o - cos_4o) - 3./2.*r_f**2 * cos_4o
                    #sigma_xy = -1.*r_f * (1./2.*sin_2o + sin_4o) + 3./2.*r_f**2 * sin_4o                

                    sxx      = lvalues_u1x[g1,g2]
                    syy      = lvalues_u2y[g1,g2]
                    sxy      = lvalues_u1y[g1,g2]
                    syx      = lvalues_u2x[g1,g2]

                    J_mat    = abs(sxx*syy-sxy*syx)
                    # ... test 0
                    uhx      = lvalues_1sx[g1,g2]
                    uhy      = lvalues_1sy[g1,g2]
                    vhx      = lvalues_2sx[g1,g2]
                    vhy      = lvalues_2sy[g1,g2]
                    
                    uamx_1   = (syy*uhx - syx*uhy)/J_mat
                    uamy_1   = (sxx*uhy - sxy*uhx)/J_mat
                    uamx_2   = (syy*vhx - syx*vhy)/J_mat
                    uamy_2   = (sxx*vhy - sxy*vhx)/J_mat
                    # ---
                    sigmah_xx = 2. * mu * uamx_1 + lanbda * (uamx_1 + uamy_2)
                    sigmah_xy = mu * (uamy_1 + uamx_2)
                    sigmah_yy = 2. * mu * uamy_2 + lanbda * (uamx_1 + uamy_2)
                    
                    wvol     = weights_1[ie1, g1] * weights_2[ie2, g2] * J_mat

                    v       += ((sigmah_xx- sigma_xx)**2)  * wvol #+ (sigmah_xy - sigma_xy)**2 + (sigmah_yy- sigma_yy)**2) * wvol
            norm_l2 += v

    norm_l2    = sqrt(norm_l2)
    rhs[p1,p2] = norm_l2    