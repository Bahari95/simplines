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
@types('int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]')
def assemble_vector_ex01(ne1, ne2, p1, p2, spans_1, spans_2,  basis_1, basis_2,  weights_1, weights_2, points_1, points_2, vector_u, rhs):

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
    lcoeffs_u  = zeros((p1+1,p2+1))
    
    
    lvalues_u  = zeros((k1, k2))
    lvalues_ux = zeros((k1, k2))
    lvalues_uy = zeros((k1, k2))        

    # ... build rhs
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

                            bj_x     = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                            bj_y     = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                            # ...
                            coeff_u   = lcoeffs_u[il_1,il_2]
                            # ...
                            sx      +=  coeff_u*bj_x
                            sy      +=  coeff_u*bj_y

                    x    = points_1[ie1, g1]
                    y    = points_2[ie2, g2]

                    # ... test 1                                 
                    #f  = 2.*x*pi**2*cos(pi*y)
                    # ... test 1                                 
                    f  = 2.*pi**2*sin(pi*x)*sin(pi*y)+sin(pi*x)*sin(pi*y)
                    
                    lvalues_u[g1,g2]  = f
                    lvalues_ux[g1,g2] = sx
                    lvalues_uy[g1,g2] = sy
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 1, g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u   = lvalues_u[g1,g2]
                            ux  = lvalues_ux[g1,g2]
                            uy  = lvalues_uy[g1,g2]
                            v += bi_0 * u * wvol - (ux * bi_x + uy * bi_y) * wvol 

                    rhs[i1+p1,i2+p2] += v   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Assembles Neumann Condition
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]        
        for il_1 in range(0, p1+1):
           i1    = i_span_1 - p1 + il_1

           vx_0 = 0.0
           vx_1 = 0.0
           for g1 in range(0, k1):
                  bi_0     =  basis_1[ie1, il_1, 0, g1]
                  wleng_x  =  weights_1[ie1, g1]
                  x1       =  points_1[ie1, g1]
                  
                  vx_0    += -1.*bi_0*pi*sin(pi*x1) * wleng_x
                  vx_1    += bi_0*pi*sin(pi*x1)*cos(pi*1.) * wleng_x

           rhs[i1+p1,0+p2]       += vx_0
           rhs[i1+p1,ne2+2*p2-1] += vx_1
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

                    # ... test 1
                    #u    = 2.*x*cos(pi*y)
                    #ux   = 2.*cos(pi*y)
                    #uy   = -2.*pi*x*sin(pi*y)
                    # ... test 1
                    u    = sin(pi*x)*sin(pi*y)
                    ux   = pi*cos(pi*x)*sin(pi*y)
                    uy   = pi*sin(pi*x)*cos(pi*y)

                    uh   = lvalues_u[g1,g2]
                    uhx  = lvalues_ux[g1,g2]
                    uhy  = lvalues_uy[g1,g2]

                    v   += (u-uh)**2 * wvol
                    w   += ((ux-uhx)**2+(uy-uhy)**2) * wvol
            norm_l2 += v
            norm_H1 += w

    norm_l2 = sqrt(norm_l2)
    norm_H1 = sqrt(norm_H1)

    rhs[p1,p2]   = norm_l2
    rhs[p1,p2+1] = norm_H1
    # ...
