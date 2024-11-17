__all__ = ['assemble_stiffnessmatrix1D',
           'assemble_massmatrix1D',
           'assemble_vector_ex01',
           'assemble_norm_ex01',
]


#---1 : In uniform mesh
from pyccel.decorators import types

# assembles stiffness matrix 1D
#==============================================================================
@types('int', 'int', 'int[:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_stiffnessmatrix1D(ne, degree, spans, basis, weights, points, matrix):

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

#==============================================================================Assembles stiffness matrix
#---1 : In uniform mesh
from pyccel.decorators import types
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:,:]')
def assemble_matrix_2D_ex01(ne1, ne2, p1, p2,  spans_1, spans_2, basis_1, basis_2, weights_1, weights_2, points_1, points_2, matrix):

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...

    # ... build matrices
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

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
                                    bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                                    bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]


                                    bj_x = basis_1[ie1, jl_1, 1, g1] * basis_2[ie2, jl_2, 0, g2]
                                    bj_y = basis_1[ie1, jl_1, 0, g1] * basis_2[ie2, jl_2, 1, g2]

                                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                                    v += (bi_x * bj_x + bi_y * bj_y) * wvol

                            matrix[p1+i1, p2+i2, p1+j1-i1, p2+j2-i2]  += v
    # ...
    
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_v, rhs):

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
    
    # ...
    lvalues_u  = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))
    # ...
    lcoeffs_v  = zeros((p1+1,p2+1))
    
    int_rhsP = 0.0
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]

            lcoeffs_v[ : , : ] = vector_v[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    #... Integration of Dirichlet boundary conditions
                    sx = 0.0
                    sy = 0.0
                    for il_1 in range(0, p1+1):
                        for il_2 in range(0, p2+1):
                            bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]
                            # ... 
                            coeff_v   = lcoeffs_v[il_1,il_2]
                            # ...
                            sx       +=  coeff_v*bi_x1
                            sy       +=  coeff_v*bi_x2
                    lvalues_ux[g1,g2] = sx
                    lvalues_uy[g1,g2] = sy

                    # ... test 1     
                    x                       = points_1[ie1, g1]
                    y                       = points_2[ie2, g2]
                    #f                = 0. 
                    # ...
                    f                = -2.*sin(5*pi*y)+25*pi**2*x**2*sin(5*pi*y) #- 10000.0*(0.5 - x)**2*exp(-50.0*(x - 0.5)**2 - 50.0*(y - 0.5)**2) - 10000.0*(0.5 - y)**2*exp(-50.0*(x - 0.5)**2 - 50.0*(y - 0.5)**2) + 200.0*exp(-50.0*(x - 0.5)**2 - 50.0*(y - 0.5)**2)

                    lvalues_u[g1,g2] = f
                    

                    
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
                            #..
                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]
                            
                            # ...Dirichlet boundary conditions
                            ux  = lvalues_ux[g1,g2]
                            uy  = lvalues_uy[g1,g2]
                            # ... rhs
                            u  = lvalues_u[g1,g2]
                            v += bi_0 * u * wvol - (ux * bi_x1  + uy * bi_x2) * wvol

                    rhs[i1+p1,i2+p2] += v   
    # ...

#==============================================================================Assemble l2 and H1 error norm
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]','double[:,:]')
def assemble_norm_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, rhs):

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
    lvalues_u = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))

    norm_l2 = 0.
    norm_H1 = 0.
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
                        b1  = basis_1[ie1,il_1,0,g1]
                        db1 = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2  = basis_2[ie2,il_2,0,g2]
                            db2 = basis_2[ie2,il_2,1,g2]

                            lvalues_u[g1,g2]  += coeff_u*b1*b2
                            lvalues_ux[g1,g2] += coeff_u*db1*b2
                            lvalues_uy[g1,g2] += coeff_u*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol = weights_1[ie1, g1] * weights_2[ie2, g2]

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]
 
                    # ...
                    u  = x**2*sin(5.0*pi*y) #+ 1.0*exp(-((x-0.5)**2 + (y-0.5)**2)/0.02)
                    ux = 2.*x* sin(5*pi*y) #+1.0*(50.0 - 100.0*x)*exp(-50.0*(x - 0.5)**2 - 50.0*(y - 0.5)**2) 
                    uy = 5*pi*x**2*cos(5*pi*y) #+ 1.0*(50.0 - 100.0*y)*exp(-50.0*(x - 0.5)**2 - 50.0*(y - 0.5)**2)
                    # ...
                    uh  = lvalues_u[g1,g2]
                    uhx = lvalues_ux[g1,g2]
                    uhy = lvalues_uy[g1,g2]

                    v  += (u-uh)**2 * wvol
                    w  += ((ux-uhx)**2+(uy-uhy)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1,p2] = norm_l2
    rhs[p1,p2+1] = norm_H1
    # ...


    #==============================================================================Assemble rhs Poisson in 3D
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:,:]', 'double[:,:,:]')
def assemble_vector_ex02(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, vector_v, rhs):

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
    lvalues_u  = zeros((k1, k2, k3))
    lvalues_ux = zeros((k1, k2, k3))
    lvalues_uy = zeros((k1, k2, k3))
    lvalues_uz = zeros((k1, k2, k3))
    lcoeffs_v  = zeros((p1+1, p2+1, p3+1))

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            for ie3 in range(0, ne3):
                i_span_3 = spans_3[ie3]

                lcoeffs_v[ : , : , : ] = vector_v[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
                for g1 in range(0, k1):
                   for g2 in range(0, k2):
                      for g3 in range(0, k3):

                         wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]*weights_3[ie3, g3]

                         #... Integration of Dirichlet boundary conditions
                         sx = 0.0
                         sy = 0.0
                         sz = 0.0
                         for il_1 in range(0, p1+1):
                            for il_2 in range(0, p2+1):
                               for il_3 in range(0, p3+1):
                                 bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                                 bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2] * basis_3[ie3, il_3, 0, g3]
                                 bi_x3 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 1, g3]
                                 # ... 
                                 coeff_v   = lcoeffs_v[il_1,il_2,il_3]
                                 # ...
                                 sx       +=  coeff_v*bi_x1
                                 sy       +=  coeff_v*bi_x2
                                 sz       +=  coeff_v*bi_x3
                         lvalues_ux[g1,g2,g3] = sx
                         lvalues_uy[g1,g2,g3] = sy
                         lvalues_uz[g1,g2,g3] = sz

                         x     =  points_1[ie1, g1]
                         y     =  points_2[ie2, g2]
                         z     =  points_3[ie3, g3]

                         #... Test 3
                         f = -2.*sin(5*pi*y)*z**2+25*pi**2*x**2*sin(5*pi*y)*z**2-2.*x**2*sin(5*pi*y)

                         lvalues_u[g1,g2,g3]  = f

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

                              bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                              bi_x1 = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 0, g3]
                              bi_x2 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2] * basis_3[ie3, il_3, 0, g3]
                              bi_x3 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2] * basis_3[ie3, il_3, 1, g3]

                              wvol  = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]

                              u     = lvalues_u[g1,g2,g3]
                              ux    = lvalues_ux[g1,g2,g3]
                              uy    = lvalues_uy[g1,g2,g3]
                              uz    = lvalues_uz[g1,g2,g3]
                              
                              v   += bi_0 * u * wvol - (ux * bi_x1  + uy * bi_x2 + uz * bi_x3) * wvol

                      rhs[i1+p1,i2+p2,i3+p3] += v   
    # ...
    
#==============================================================================Assemble l2 and H1 error norm in 3D
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:,:]', 'double[:,:,:]')
def assemble_norm_ex02(ne1, ne2, ne3, p1, p2, p3, spans_1, spans_2, spans_3,  basis_1, basis_2, basis_3,  weights_1, weights_2, weights_3, points_1, points_2, points_3, vector_u, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    k3 = weights_3.shape[1]
    # ...

    lcoeffs_u = zeros((p1+1,p2+1,p3+1))
    lvalues_u = zeros((k1, k2, k3))
    lvalues_ux = zeros((k1, k2, k3))
    lvalues_uy = zeros((k1, k2, k3))
    lvalues_uz = zeros((k1, k2, k3))

    norm_l2 = 0.
    norm_H1 = 0.
    # ...
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            for ie3 in range(0, ne3):
               i_span_3 = spans_3[ie3]

               lvalues_u [ : , : , : ] = 0.0
               lvalues_ux[ : , : , : ] = 0.0
               lvalues_uy[ : , : , : ] = 0.0
               lvalues_uz[ : , : , : ] = 0.0
               lcoeffs_u [ : , : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1, i_span_3 : i_span_3+p3+1]
               for il_1 in range(0, p1+1):
                  for il_2 in range(0, p2+1):
                     for il_3 in range(0, p3+1):
                         coeff_u = lcoeffs_u[il_1, il_2, il_3]

                         for g1 in range(0, k1):
                            b1   = basis_1[ie1,il_1,0,g1]
                            db1  = basis_1[ie1,il_1,1,g1]
                            
                            for g2 in range(0, k2):
                               b2   = basis_2[ie2,il_2,0,g2]
                               db2  = basis_2[ie2,il_2,1,g2]
                               
                               for g3 in range(0, k3):
                                   b3   = basis_3[ie3,il_3,0,g3]
                                   db3  = basis_3[ie3,il_3,1,g3]

                                   lvalues_u [g1,g2,g3] += coeff_u * b1*b2*b3
                                   lvalues_ux[g1,g2,g3] += coeff_u * db1*b2*b3
                                   lvalues_uy[g1,g2,g3] += coeff_u * b1*db2*b3
                                   lvalues_uz[g1,g2,g3] += coeff_u * b1*b2*db3

               v = 0.0
               w = 0.0
               for g1 in range(0, k1):
                  for g2 in range(0, k2):
                     for g3 in range(0, k3):
                        wvol = weights_1[ie1, g1] * weights_2[ie2, g2] * weights_3[ie3, g3]

                        x1   = points_1[ie1, g1]
                        x2   = points_2[ie2, g2]
                        x3   = points_3[ie3, g3]

                        # ... test 1
                        #ux   = 2/3.*x1*exp((x1**2+x2**2+x3**2)/3.)
                        #uy   = 2/3.*x2*exp((x1**2+x2**2+x3**2)/3.)
                        #uz   = 2/3.*x3*exp((x1**2+x2**2+x3**2)/3.)
                        # ... test 2
                        u    = x1**2*sin(5.*pi*x2)*x3**2  
                        ux   = 2.*x1*sin(5.*pi*x2)*x3**2  
                        uy   = x1**2*5.*pi*cos(5.*pi*x2)*x3**2  
                        uz   = x1**2*sin(5.*pi*x2)*x3*2.  
                        
                        uh   = lvalues_u[g1,g2,g3]
                        uhx  = lvalues_ux[g1,g2,g3]
                        uhy  = lvalues_uy[g1,g2,g3]
                        uhz  = lvalues_uz[g1,g2,g3]

                        v  += (u-uh)**2 * wvol
                        w  += ((ux-uhx)**2 + (uy-uhy)**2 + (uz-uhz)**2) * wvol
               norm_l2 += v
               norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1,p2, p3]   = norm_l2
    rhs[p1,p2, p3+1] = norm_H1
    # ...
    
#==============================================================================
#
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_ex20(ne1, ne3, p1, p3, spans_1, spans_3,  basis_1, basis_3,  weights_1, weights_3, points_1, points_3, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k3 = weights_3.shape[1]
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie3 in range(0, ne3):
               i_span_3 = spans_3[ie3]
               
               for il_1 in range(0, p1+1):
                     for il_3 in range(0, p3+1):
                   
                        i1 = i_span_1 - p1 + il_1
                        i3 = i_span_3 - p3 + il_3

                        v = 0.0
                        for g1 in range(0, k1):
                              for g3 in range(0, k3):
        
                                bi_0  = basis_1[ie1, il_1, 0, g1]* basis_3[ie3,il_3,0,g3]
                                #...
                                wvol  = weights_1[ie1, g1] * weights_3[ie3, g3]
                                #..
                                x1    = points_1[ie1, g1]
                                x3    = points_3[ie3, g3]
                                #__
                                u     = x1**2*sin(5.*pi*(-1.))*x3**2         
                                #..
                                v += bi_0 * u * wvol

                        rhs[i1+p1, i3+p3] += v 
                        
#==============================================================================
#
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]')
def assemble_vector_ex21(ne1, ne3, p1, p3, spans_1, spans_3,  basis_1, basis_3,  weights_1, weights_3, points_1, points_3, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import zeros

    # ... sizes
    k1 = weights_1.shape[1]
    k3 = weights_3.shape[1]
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie3 in range(0, ne3):
               i_span_3 = spans_3[ie3]
               
               for il_1 in range(0, p1+1):
                     for il_3 in range(0, p3+1):
                   
                        i1 = i_span_1 - p1 + il_1
                        i3 = i_span_3 - p3 + il_3

                        v = 0.0
                        for g1 in range(0, k1):
                              for g3 in range(0, k3):
        
                                bi_0  = basis_1[ie1, il_1, 0, g1]* basis_3[ie3,il_3,0,g3]
                                #...
                                wvol  = weights_1[ie1, g1] * weights_3[ie3, g3]
                                #..
                                x1    = points_1[ie1, g1]
                                x3    = points_3[ie3, g3]
                                #__
                                u     = x1**2*sin(5.*pi*1.)*x3**2       
                                #..
                                v += bi_0 * u * wvol

                        rhs[i1+p1, i3+p3] += v   
    # ...