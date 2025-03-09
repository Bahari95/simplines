__all__ = ['assemble_vector_ex01',
           'assemble_norm_ex01'
]
from pyccel.decorators import types



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
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_x = basis[ie1, il_1, 1, g1]
                                    bj_x = basis[ie1, il_2, 1, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_x * bj_x * wvol

                            matrix[ degree+ i1, degree+ i2-i1]  += v

# assembles mass matrix 1D
#==============================================================================
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
                            v = 0.0
                            for g1 in range(0, k1):
                                
                                    bi_0 = basis[ie1, il_1, 0, g1]
                                    bj_0 = basis[ie1, il_2, 0, g1]
                                    
                                    wvol = weights[ie1, g1]
                                    
                                    v += bi_0 * bj_0 * wvol

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
                                    
                                    v += bi_x * bj_0 * wvol

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
                                    
                                    v += bi_0 * bj_x * wvol

                            matrix[i1+p1,i2+p2]  += v
    # ...

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, rhs):

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
    lvalues_u = zeros((k1, k2))

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lvalues_u[ : , : ] = 0.0

            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]
                                                          
                    # ... test 1                                 
                    #f  = -pi**2*x**2*cos(pi*y) + 2*cos(pi*y)
                    # ... test 2                                 
                    f  = 4.*x**2*exp(1.-x**2)*cos(pi*y**2) - 4.*pi**2*y**2*exp(1.-x**2)*cos(pi*y**2) - 2.*pi*exp(1.-x**2)*sin(pi*y**2) - 2.*exp(1.-x**2)*cos(pi*y**2)
                    # ... test 3                                 
                    #f  = -4*pi**2*x**2*y**2*(1.0 - x)**2*cos(pi*y**2) - 2*pi*x**2*(1.0 - x)**2*sin(pi*y**2) + 2*x**2*cos(pi*y**2) + 4*x*(2*x - 2.0)*cos(pi*y**2) + 2*(1.0 - x)**2*cos(pi*y**2)

                    lvalues_u[g1,g2] = f
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, vector_w, rhs):

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

    lcoeffs_u = zeros((p1+1,p2+1))
    lcoeffs_w = zeros((p2+1,p1+1))
    #lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    #norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_2[ie1]
        for ie2 in range(0, ne2):
            i_span_3 = spans_2[ie2]
            i_span_2 = spans_1[ie2]

            lvalues_ux[ : , : ] = 0.0
            lcoeffs_u[ : , : ]  = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p2+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_1[ie1,il_1,0,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2] #M^p2-1

                            lvalues_ux[g1,g2] += coeff_u*b1*b2
            lvalues_uy[ : , : ] = 0.0
            lcoeffs_w[ : , : ]  = vector_w[i_span_4 : i_span_4+p2+1, i_span_2 : i_span_2+p1+1]
            for il_1 in range(0, p2+1):
                for il_2 in range(0, p1+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1  = basis_2[ie1,il_1,0,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2  = basis_1[ie2,il_2,0,g2] 

                            lvalues_uy[g1,g2] += coeff_w*b1*b2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]

                    # ... test 1
                    #u  =  sin(pi*x1)*sin(pi*x2)
                    #ux =  2*x*cos(pi*y)
                    #uy =  -pi*x**2*sin(pi*y)
                    # ... test 2
                    #u  =   exp((1-2.*x)**2)*cos(pi*y**2)
                    ux =  -2.*x*exp(1.-x**2)*cos(pi*y**2)
                    uy =  -2.*pi*y*exp(1.-x**2)*sin(pi*y**2)
                    # ... test 3
                    #u  =  (x*(1.-x))**2*cos(pi*y**2)
                    #ux =  x**2*(2*x - 2.0)*cos(pi*y**2) + 2*x*(1.0 - x)**2*cos(pi*y**2)
                    #uy =  -2*pi*x**2*y*(1.0 - x)**2*sin(pi*y**2)

                    #uh  = lvalues_u[g1,g2]
                    uhx = lvalues_ux[g1,g2]
                    uhy = lvalues_uy[g1,g2]

                    #v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2 + (uy-uhy)**2) * wvol
            #norm_l2 += v
            norm_H1 += w

    #norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    #rhs[p1,p2] = norm_l2
    rhs[p1,p2+1] = norm_H1
    # ...
