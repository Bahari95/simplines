__all__ = ['assemble_matrix_ex03',
           'assemble_vector_ex03',
           'assemble_massmatrix1D',
           'assemble_vector_ex01',
           'assemble_norm_ex01'
]


#__________________________________
from pyccel.decorators import types

#==============================================================================
#---2 : In adapted mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real', 'int', 'double[:,:,:,:]')
def assemble_matrix_ex03(ne1, ne2, p1, p2,  spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, vector_u, dt, alpha, matrix):

    # ... sizes
    from numpy import zeros
    k1         = weights_1.shape[1]
    k2         = weights_2.shape[1]
    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    # ...
    arr_u      = zeros((k1,k2))
    arr_ux     = zeros((k1,k2))
    arr_uy     = zeros((k1,k2))
    #...
    arr_uxx    = zeros((k1,k2))
    arr_uyy    = zeros((k1,k2))
    #...
    theta      = 3./2    
    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            # ...
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    s   = 0.0
                    sx  = 0.0
                    sy  = 0.0
                    
                    sxx = 0.0
                    syy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):

                            bj_0      = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                            bj_x      = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y      = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            
                            bj_xx     = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                            bj_yy     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                            # ...
                            coeff_u   = lcoeffs_u[il_1,il_2]
                            # ...
                            s        +=  coeff_u*bj_0
                            sx       +=  coeff_u*bj_x
                            sy       +=  coeff_u*bj_y
                            # ...
                            sxx      +=  coeff_u*bj_xx
                            syy      +=  coeff_u*bj_yy

                    arr_u[g1,g2]      = s
                    arr_ux[g1,g2]     = sx
                    arr_uy[g1,g2]     = sy
                    
                    arr_uxx[g1,g2]    = syy
                    arr_uyy[g1,g2]    = sxx

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    for jl_1 in range(0, p1+1):
                        for jl_2 in range(0, p2+1):
                            i1 = i_span_1 - p1 + il_1
                            j1 = i_span_1 - p1 + jl_1
                            # ...
                            i2 = i_span_2 - p2 + il_2
                            j2 = i_span_2 - p2 + jl_2
                            # ...
                            v = 0.0
                            for g1 in range(0, k1):
                                for g2 in range(0, k2):
                                    bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                                    bi_xx = basis_1[ie1, il_1, 2, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_yy = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 2, g2]
                                    
                                    #........
                                    bj_0  = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x1 = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_x2 = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    bj_xx = basis_1[ie1, jl_1, 2, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_yy = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 2, g2]
                                    #........ 
                                    u    = arr_u[g1,g2] 
                                    ux   = arr_ux[g1,g2]
                                    uy   = arr_uy[g1,g2]
                    
                                    uxx  = arr_uxx[g1,g2]
                                    uyy  = arr_uyy[g1,g2]                                    
                                    #..
                                    R_1  = ( (3.*alpha/(2.*theta))*(1.-4.*theta*u*(1.-u)) + (1.-2.*u)*(uxx+uyy) ) * (bj_x1*bi_x1 + bj_x2 * bi_x2)
                                    R_2  = ( (-6.*alpha)*(1.-2.*u)*bi_0 - 2.*bi_0*(uxx+uyy) + (1.-2.*u)*(bi_xx+bi_yy) ) * (bj_x1*ux + bj_x2 * uy)
                                    #..
                                    R_3  = (bj_xx+bj_yy)*(bi_xx+bi_yy)*u*(1.-u)
                                    R_4  = (bj_xx+bj_yy)*(uxx+uyy)*(1.-2.*u)*bi_0
                                    
                                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v    +=  bi_0 * bj_0 * wvol +  dt * ( R_1 + R_2 + R_3 + R_4 ) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...

#==============================================================================
#---
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real', 'int', 'double[:,:]')
def assemble_vector_ex03(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w, dt, alpha, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import arctan2
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    #...
    theta      = 3./2
    # ...
    lcoeffs_u  = zeros((p1+1,p2+1))
    lcoeffs_w  = zeros((p1+1,p2+1))
    # ...
    arr_ut     = zeros((k1,k2))
    arr_u      = zeros((k1,k2))
    arr_ux     = zeros((k1,k2))
    arr_uy     = zeros((k1,k2))
    # ...
    arr_uxx    = zeros((k1,k2))
    arr_uyy    = zeros((k1,k2))
    # +++
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_w[ : , : ] = vector_w[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    st  = 0.0
                    #...
                    s   = 0.0
                    sx  = 0.0
                    sy  = 0.0
                    sxx = 0.0
                    syy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]
                              bj_x    = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y    = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              # +++
                              bj_xx   = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy   = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              # +++
                              coeff_u = lcoeffs_u[il_1,il_2]
                              # +++
                              s      +=  coeff_u*bj_0
                              sx     +=  coeff_u*bj_x
                              sy     +=  coeff_u*bj_y
                              
                              sxx    +=  coeff_u*bj_xx
                              syy    +=  coeff_u*bj_yy
                              # +++
                              coeff_w = lcoeffs_w[il_1,il_2]
                              # +++
                              st     +=  coeff_w*bj_0


                    arr_ut[g1,g2]     = st
                    arr_u[g1,g2]      = s
                    arr_ux[g1,g2]     = sx
                    arr_uy[g1,g2]     = sy
                    
                    arr_uxx[g1,g2]    = syy
                    arr_uyy[g1,g2]    = sxx
                    
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):

                            bi_0    = basis_1[ie1, il_1,0,g1] * basis_2[ie2,il_2,0,g2]
                            bi_x    = basis_1[ie1, il_1,1,g1] * basis_2[ie2,il_2,0,g2]
                            bi_y    = basis_1[ie1, il_1,0,g1] * basis_2[ie2,il_2,1,g2]
                            # +++
                            bi_xx   = basis_1[ie1, il_1,2,g1] * basis_2[ie2,il_2,0,g2]
                            bi_yy   = basis_1[ie1, il_1,0,g1] * basis_2[ie2,il_2,2,g2]
                            
                            wvol    = weights_1[ie1, g1] * weights_2[ie2, g2]
                            #........ 
                            ut      = arr_ut[g1,g2] 
                            u       = arr_u[g1,g2] 
                            ux      = arr_ux[g1,g2]
                            uy      = arr_uy[g1,g2]
                    
                            uxx     = arr_uxx[g1,g2]
                            uyy     = arr_uyy[g1,g2] 
                            #...
                            R_1     = ((3.*alpha/(2.*theta))*(1. - 4.*theta*u*(1.-u) ) + (1.-2.*u) * (uxx+uyy) ) * (bi_x * ux + bi_y * uy) 
                            
                            R_2     = u * (1. - u) * (uxx + uyy) * (bi_xx + bi_yy) 

                            v      +=  bi_0 * u * wvol + dt* (R_1 * wvol + R_2 * wvol) - 1.* bi_0 * ut * wvol

                    rhs[i1+p1,i2+p2] += v
    # ...

# assembles mass matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_massmatrix1D(ne, degree, spans, basis, weights, points,  matrix):

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
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_0 * bj_0 * wvol

                            matrix[degree+i1, degree+ i2-i1]  += v


#=============================================================================
#---
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, rhs):

    from numpy import zeros
    from numpy import random
    from numpy import sin
    from numpy import exp
    from numpy import pi
    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lvalues_u  = zeros((k1, k2))
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            f                = (2.*random.rand()-1.)*0.05 +0.63
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    x = points_1[ie1, g1]
                    y = points_2[ie2, g2]
                    lvalues_u[g1,g2] = f
                    
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            #..
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            # ... rhs
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol 

                    rhs[i1+p1,i2+p2] += v   
    # ...
#==============================================================================
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'int', 'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, alpha, rhs):

    from numpy import sin
    from numpy import cos
    from numpy import pi
    from numpy import sqrt
    from numpy import exp
    from numpy import log
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    theta       = 3./2
    # ...
    t = 0.
    lcoeffs_u  = zeros((p1+1,p2+1))
    lvalues_u  = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    GL_free_energy = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ]  = 0.0
            lvalues_ux[ : , : ] = 0.0
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]
                    for g1 in range(0, k1):
                        b1 = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_2[ie2,il_2,0,g2]
                            db2  = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]  += coeff_u*b1*b2
                            lvalues_ux[g1,g2] += coeff_u*db1*b2
                            lvalues_uy[g1,g2] += coeff_u*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    
                    #+--------
                    uh   = lvalues_u[g1,g2]
                    uhx  = lvalues_ux[g1,g2]
                    uhy  = lvalues_uy[g1,g2]
                    #+++++++++
                    if uh > 1. :
                        uh = 0.999999
                    if uh < 0. :
                         uh = 0.00001    
                    w   += ( uh*log(uh) + (1.-uh)*log(1.-uh) + 2.*theta*uh*(1.-uh)+ theta/(3.*alpha)*(uhx**2 + uhy**2) )* wvol
            GL_free_energy += w

    rhs[p1, p2]   = GL_free_energy
    # ...
