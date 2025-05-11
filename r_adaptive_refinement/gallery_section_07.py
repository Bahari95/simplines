__all__ = ['assemble_vector_ex01',
           'assemble_norm_ex01'
]
from pyccel.decorators import types


#==============================================================================
# Assembles a rhs at t_0
#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'float', 'double[:,:]')
def assemble_vector_ex00(ne1, ne2, ne3, ne4, p1, p2, p3, p4, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, weights_1, weights_2, weights_3, weights_4, points_1, points_2, points_3, points_4, vector_v1, vector_v2, epsilon, gamma, dt, times, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty

    # ... sizes
    k1 = weights_1.shape[1]
    k2 = weights_2.shape[1]
    # ...
    lcoeffs_v1  = zeros((p3+1,p4+1))
    lcoeffs_v2  = zeros((p3+1,p4+1))
    
    # ...coefficient of normalisation
    crho      = 0.0
    for ie1 in range(0, ne1):
        i_span_3 = spans_3[ie1]
        for ie2 in range(0, ne2):
            i_span_4 = spans_4[ie2]
            
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                    #x    = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    #y    = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    x         = 0.0
                    y         = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              
                              bi_0      = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]
                              # ...
                              x       +=  coef_v1*bi_0
                              y       +=  coef_v2*bi_0
                    
                    #.. Test 1
                    #rho  = 1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4)) 
                    rho  = 2.+sin(4.*pi*sqrt((x-0.6)**2+(y-0.6)**2)) 
                    #rho  = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2) 
                    #rho   = 1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2
                    # .. butterfluy
                    #rho   = 1.+7.*exp(-50.*abs((x)**2+(y-0.25*times)**2-0.09))  
                    crho += rho * wvol
    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_3 = spans_3[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_4 = spans_4[ie2]
            
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0  = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_x  = basis_1[ie1, il_1, 1, g1] * basis_2[ie2, il_2, 0, g2]
                            bi_y  = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            x         = 0.0
                            y         = 0.0
                            # ..
                            for jl_1 in range(0, p3+1):
                                  for jl_2 in range(0, p4+1):
                                      coef_v1   = lcoeffs_v1[jl_1,jl_2]
                                      coef_v2   = lcoeffs_v2[jl_1,jl_2]
                                      bi_0      = basis_3[ie1,jl_1,0,g1]*basis_4[ie2,jl_2,0,g2]
                                      # ...
                                      x       +=  coef_v1*bi_0
                                      y       +=  coef_v2*bi_0
                            #.. Test 1
                            #rho  = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*0.)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.7-0.25*0.) -20.*((x-0.7-0.25*0.)**2+(y-0.5)**2)))**2)
                            #rho  = 1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4))         
                            # .. circle
                            #rho   = 1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2
                            # ... quarter-annulus
                            rho  = 2.+sin(4.*pi*sqrt((x-0.6)**2+(y-0.6)**2))  
                            # .. butterfluy
                            #rho   = 1.+7.*exp(-50.*abs((x)**2+(y-0.25*times)**2-0.09))  
                    
                            sx     = points_1[ie1, g1]
                            sy     = points_2[ie2, g2]
                            # ...
                            rho       = rho/crho          
                            u         = (sx**2+sy**2)*0.5
                            laplace_u = 2.0
                            mae_rho   = sqrt(rho)

                            v += ( epsilon*(u - gamma*2.0) + dt*mae_rho)* bi_0 * wvol

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
                  
                  vx_0    += bi_0*0.0 * wleng_x
                  vx_1    += bi_0*1. * wleng_x

           rhs[i1+p1,0+p2]       += epsilon*gamma*vx_0
           rhs[i1+p1,ne2+2*p2-1] += epsilon*gamma*vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  wleng_y =  weights_2[ie2, g2]
                  x2      =  points_2[ie2, g2]
                           
                  vy_0   += bi_0* 0.0 * wleng_y
                  vy_1   += bi_0*1. * wleng_y

           rhs[0+p1,i2+p2]       += epsilon*gamma*vy_0
           rhs[ne1-1+2*p1,i2+p2] += epsilon*gamma*vy_1
    # ...

    

#==============================================================================Assemble rhs Poisson
#---1 : In uniform mesh
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int', 'int[:]', 'int[:]', 'int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'float', 'float', 'float', 'float', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:]')
def assemble_vector_ex01(ne1, ne2, ne3, ne4, p1, p2, p3, p4, spans_1, spans_2,  spans_3, spans_4, basis_1, basis_2, basis_3, basis_4, weights_1, weights_2, weights_3, weights_4, points_1, points_2, points_3, points_4, vector_u, vector_v1, vector_v2, epsilon, gamma, dt, times, spans_ad1, spans_ad2, basis_ad1, basis_ad2, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty
    # ... sizes
    k1          = weights_1.shape[1]
    k2          = weights_2.shape[1]
    lcoeffs_u   = zeros((p1+1,p2+1))
    lcoeffs_v1  = zeros((p3+1,p4+1))
    lcoeffs_v2  = zeros((p3+1,p4+1))

    lvalues_u   = zeros((k1, k2))
    lvalues_D   = zeros((k1, k2))
    lvalues_rho = zeros((k1, k2))
    # ...
    # ...coefficient of normalisation
    crho      = 0.0
    for ie1 in range(0, ne1):
        i_span_3 = spans_3[ie1]
        for ie2 in range(0, ne2):
            i_span_4 = spans_4[ie2]
            
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_3 : i_span_3+p3+1, i_span_4 : i_span_4+p4+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1] * weights_2[ie2, g2]

                    #x    = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    #y    = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    x         = 0.0
                    y         = 0.0
                    # ..
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              
                              bi_0      = basis_3[ie1,il_1,0,g1]*basis_4[ie2,il_2,0,g2]
                              
                              # ...
                              x        +=  coef_v1*bi_0
                              y        +=  coef_v2*bi_0
                    
                    #.. Test 1
                    #rho  = 1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4)) 
                    rho  = 2.+sin(4.*pi*sqrt((x-0.6-0.25*times)**2+(y-0.6)**2)) 
                    #rho = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*times)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.7-0.25*times) -20.*((x-0.7-0.25*times)**2+(y-0.5)**2)))**2) 
                    #rho   = 1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2
                    # .. butterfluy
                    #rho   = 1.+7.*exp(-50.*abs((x)**2+(y-0.25*times)**2-0.09))  
                    crho += rho * wvol

    # ... build rhs
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            
            lcoeffs_u[ : , : ] = vector_u[i_span_1 : i_span_1+p1+1, i_span_2 : i_span_2+p2+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    s   = 0.0
                    x1  = 0.0
                    x2  = 0.0
                    sxx = 0.0
                    syy = 0.0
                    sxy = 0.0
                    for il_1 in range(0, p1+1):
                          for il_2 in range(0, p2+1):

                              bj_0  = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,0,g2]

                              bj_x  = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,0,g2]
                              bj_y  = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,1,g2]
                              
                              bj_xx = basis_1[ie1,il_1,2,g1]*basis_2[ie2,il_2,0,g2]
                              bj_yy = basis_1[ie1,il_1,0,g1]*basis_2[ie2,il_2,2,g2]
                              bj_xy = basis_1[ie1,il_1,1,g1]*basis_2[ie2,il_2,1,g2]

                              coeff_u = lcoeffs_u[il_1,il_2]

                              s    +=  coeff_u*bj_0

                              x1   +=  coeff_u*bj_x
                              x2   +=  coeff_u*bj_y
                              
                              sxx  +=  coeff_u*bj_xx
                              syy  +=  coeff_u*bj_yy
                              sxy  +=  coeff_u*bj_xy
                    # ...
                    span_3 = spans_ad1[ie1, ie2, g1, g2]
                    span_4 = spans_ad2[ie1, ie2, g1, g2]

                    #------------------   
                    lcoeffs_v1[ : , : ]  =  vector_v1[span_3 : span_3+p3+1, span_4 : span_4+p4+1]
                    lcoeffs_v2[ : , : ]  =  vector_v2[span_3 : span_3+p3+1, span_4 : span_4+p4+1]
                    
                    #x     = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    #y     = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    x      = 0.0
                    y      = 0.0
                    for il_1 in range(0, p3+1):
                          for il_2 in range(0, p4+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              bi_0      = basis_ad1[ie1, ie2,il_1,0, g1, g2]*basis_ad2[ie1, ie2,il_2,0, g1, g2]
                              # ...
                              x        +=  coef_v1*bi_0
                              y        +=  coef_v2*bi_0
                    #.. Test 1
                    #rho  = 1.+5.*exp(-100.*abs((x-0.45)**2+(y-0.4)**2-0.1))+5.*exp(-100.*abs(x**2+y**2-0.2))+5.*exp(-100*abs((x+0.45)**2 +(y-0.4)**2-0.1)) +7.*exp(-100.*abs(x**2+(y+1.25)**2-0.4)) 
                    rho  = 2.+sin(4.*pi*sqrt((x-0.6-0.25*times)**2+(y-0.6)**2)) 
                    #rho = 1.+ 9./(1.+(10.*sqrt((x-0.7-0.25*times)**2+(y-0.5)**2)*cos(arctan2(y-0.5,x-0.7-0.25*times) -20.*((x-0.7-0.25*times)**2+(y-0.5)**2)))**2) 
                    #rho   = 1. + 5./cosh( 5.*((x-sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((x+sqrt(3)/2)**2+(y-0.5)**2 - (pi/2)**2) )**2
                    # .. butterfluy
                    #rho   = 1.+7.*exp(-50.*abs((x)**2+(y-0.25*times)**2-0.09)) 

                    #...
                    rho                    = rho/crho
                    lvalues_u[g1,g2]       = s
                    lvalues_D[g1,g2]       = sxx+syy
                    lvalues_rho[g1,g2]     = sqrt(rho*abs(sxx*syy-sxy**2))

            for il_1 in range(0, p1+1):
                for il_2 in range(0, p2+1):
                    i1 = i_span_1 - p1 + il_1
                    i2 = i_span_2 - p2 + il_2

                    v = 0.0
                    for g1 in range(0, k1):
                        for g2 in range(0, k2):
                            bi_0 = basis_1[ie1, il_1, 0, g1] * basis_2[ie2, il_2, 0, g2]

                            wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                            u         = lvalues_u[g1,g2]
                            mae_rho   = lvalues_rho[g1,g2]
                            laplace_u = lvalues_D[g1,g2]

                            v += bi_0 *( epsilon*(u - gamma*laplace_u) + dt*mae_rho )* wvol

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
                  
                  vx_0    += bi_0*0.0 * wleng_x
                  vx_1    += bi_0*1. * wleng_x

           rhs[i1+p1,0+p2]       += epsilon*gamma*vx_0
           rhs[i1+p1,ne2+2*p2-1] += epsilon*gamma*vx_1
    for ie2 in range(0, ne2):
        i_span_2 = spans_2[ie2]        
        for il_2 in range(0, p2+1):
           i2    = i_span_2 - p2 + il_2

           vy_0 = 0.0
           vy_1 = 0.0
           for g2 in range(0, k2):
                  bi_0    =  basis_2[ie2, il_2, 0, g2]
                  wleng_y =  weights_2[ie2, g2]
                  x2      =  points_2[ie2, g2]
                           
                  vy_0   += bi_0* 0.0 * wleng_y
                  vy_1   += bi_0*1. * wleng_y

           rhs[0+p1,i2+p2]       += epsilon*gamma*vy_0
           rhs[ne1-1+2*p1,i2+p2] += epsilon*gamma*vy_1
    # ...


# Assembles Quality of mesh adaptation
#==============================================================================
@types('int', 'int', 'int', 'int', 'int', 'int', 'int', 'int','int', 'int', 'int', 'int', 'int[:]', 'int[:]','int[:]', 'int[:]','int[:]', 'int[:]', 'double[:,:,:,:]', 'double[:,:,:,:]',  'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:,:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]',  'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'double[:,:]', 'real', 'int[:,:,:,:]', 'int[:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:,:,:,:,:]', 'double[:,:]')
def assemble_Quality_ex01(ne1, ne2, ne3, ne4, ne5, ne6, p1, p2, p3, p4, p5, p6, spans_1, spans_2,  spans_3, spans_4, spans_5, spans_6, basis_1, basis_2, basis_3, basis_4, basis_5, basis_6, weights_1, weights_2, weights_3, weights_4, weights_5, weights_6, points_1, points_2, points_3, points_4, points_5, points_6, vector_u, vector_w, vector_v1, vector_v2, times, spans_ad1, spans_ad2, basis_ad1, basis_ad2, rhs):

    from numpy import exp
    from numpy import cos
    from numpy import sin
    from numpy import pi
    from numpy import arctan2
    from numpy import sqrt
    from numpy import cosh
    from numpy import zeros
    from numpy import empty
    # ... sizes
    k1           = weights_1.shape[1]
    k2           = weights_2.shape[1]
    # ...
    lcoeffs_u    = zeros((p1+1,p3+1))
    lcoeffs_w    = zeros((p4+1,p2+1))
    lvalues_u    = zeros((k1, k2))
    # ...
    lvalues_u1   = zeros((k1, k2))
    lvalues_u1x  = zeros((k1, k2))
    lvalues_u1y  = zeros((k1, k2))
    lvalues_u2   = zeros((k1, k2))
    #lvalues_u2x = zeros((k1, k2))
    lvalues_u2y  = zeros((k1, k2))
    lcoeffs_v1  = zeros((p5+1,p6+1))
    lcoeffs_v2  = zeros((p5+1,p6+1))

    # ... build rhs
    # ...coefficient of normalisation
    Crho      = 0.0
    for ie1 in range(0, ne1):
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_6 = spans_6[ie2]
            
            lcoeffs_v1[ : , : ]  =  vector_v1[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            lcoeffs_v2[ : , : ]  =  vector_v2[i_span_5 : i_span_5+p5+1, i_span_6 : i_span_6+p6+1]
            for g1 in range(0, k1):
                for g2 in range(0, k2):

                    wvol  = weights_1[ie1, g1]*weights_2[ie2, g2]

                    #sx    = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    #sy    = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    sx         = 0.0
                    sy         = 0.0
                    for il_1 in range(0, p5+1):
                          for il_2 in range(0, p6+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              
                              bi_0      = basis_5[ie1,il_1,0,g1]*basis_6[ie2,il_2,0,g2]
                              # ...
                              sx        +=  coef_v1*bi_0
                              sy        +=  coef_v2*bi_0
                    
                    #.. Test 1
                    #rho  = 1.+ 9./(1.+(10.*sqrt((sx-0.7-0.25*0.)**2+(sy-0.5)**2)*cos(arctan2(sy-0.5,sx-0.7-0.25*0.) -20.*((sx-0.7-0.25*0.)**2+(sy-0.5)**2)))**2)
                    #rho   = 1.+5.*exp(-100.*abs((sx-0.45)**2+(sy-0.4)**2-0.1))+5.*exp(-100.*abs(sx**2+sy**2-0.2))+5.*exp(-100*abs((sx+0.45)**2 +(sy-0.4)**2-0.1))  +7.*exp(-100.*abs(sx**2+(sy+1.25)**2-0.4))
                    #.. Test 2 
                    #rho  = 1./(2.+cos(4.*pi*sqrt((sx-0.5-0.25*0.)**2+(sy-0.5)**2)))
                    rho  = 2.+sin(4.*pi*sqrt((sx-0.6)**2+(sy-0.6)**2)) 
                    #.. Test 3 
                    #rho  =  1.+ 3./(1.+(10.*sqrt(sx**2+sy**2)*cos(arctan2(sy,sx) -20.*(sx**2+sy**2)))**2)
                    #.. Test 4 
                    #rho  = 1. + 5./cosh( 5.*((sx-sqrt(3)/2)**2+(sy-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((sx+sqrt(3)/2)**2+(sy-0.5)**2 - (pi/2)**2) )**2
                    # .. butterfly
                    #rho   = 1.+7.*exp(-50.*abs((sx)**2+(sy-0.25*times)**2-0.09))
                    Crho += rho * wvol
                    
    Qual_l2      = 0.                                
    displacement = 0.
    min_det      = 1e100
    for ie1 in range(0, ne1):
        i_span_1 = spans_1[ie1]
        i_span_4 = spans_4[ie1]
        i_span_5 = spans_5[ie1]
        for ie2 in range(0, ne2):
            i_span_2 = spans_2[ie2]
            i_span_3 = spans_3[ie2]
            i_span_6 = spans_6[ie2]

            lvalues_u1[ : , : ]  = 0.0
            lvalues_u1x[ : , : ] = 0.0
            lvalues_u1y[ : , : ] = 0.0
            lcoeffs_u[ : , : ]   = vector_u[i_span_1 : i_span_1+p1+1, i_span_3 : i_span_3+p3+1]
            for il_1 in range(0, p1+1):
                for il_2 in range(0, p3+1):
                    coeff_u = lcoeffs_u[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_1[ie1,il_1,0,g1]
                        db1  = basis_1[ie1,il_1,1,g1]
                        for g2 in range(0, k2):
                            b2   = basis_3[ie2,il_2,0,g2]  #M^p2-1
                            db2  = basis_3[ie2,il_2,1,g2]  #M^p2-1

                            lvalues_u1[g1,g2]  += coeff_u*b1*b2
                            lvalues_u1x[g1,g2] += coeff_u*db1*b2
                            lvalues_u1y[g1,g2] += coeff_u*b1*db2
            lvalues_u2[ : , : ]  = 0.0
            lvalues_u2y[ : , : ] = 0.0

            lcoeffs_w[ : , : ] = vector_w[i_span_4 : i_span_4+p4+1, i_span_2 : i_span_2+p2+1]
            for il_1 in range(0, p4+1):
                for il_2 in range(0, p2+1):
                    coeff_w = lcoeffs_w[il_1,il_2]

                    for g1 in range(0, k1):
                        b1   = basis_4[ie1,il_1,0,g1] #M^p1-1
                        for g2 in range(0, k2):
                            b2   = basis_2[ie2,il_2,0,g2] 
                            db2  = basis_2[ie2,il_2,1,g2] 
                            lvalues_u2[g1,g2]  += coeff_w*b1*b2
                            lvalues_u2y[g1,g2] += coeff_w*b1*db2

            v = 0.0
            w = 0.0
            for g1 in range(0, k1):
                for g2 in range(0, k2):
                    wvol   = weights_1[ie1, g1] * weights_2[ie2, g2]
                    x     =  points_1[ie1, g1]
                    y     =  points_2[ie2, g2]

                    x1     = lvalues_u1[g1,g2]
                    x2     = lvalues_u2[g1,g2]
                    #... We compute firstly the span in new adapted points
                    span_5 = spans_ad1[ie1, ie2, g1, g2]
                    span_6 = spans_ad2[ie1, ie2, g1, g2]

                    #------------------   
                    lcoeffs_v1[ : , : ]  =  vector_v1[span_5 : span_5+p5+1, span_6 : span_6+p6+1]
                    lcoeffs_v2[ : , : ]  =  vector_v2[span_5 : span_5+p5+1, span_6 : span_6+p6+1]
                    #sx     = (2.0*x1-1.0)*sqrt(1.0-0.5*(2.0*x2-1.0)**2)
                    #sy     = (2.0*x2-1.0)*sqrt(1.0-0.5*(2.0*x1-1.0)**2)
                    sx     = 0.0
                    sy     = 0.0
                    for il_1 in range(0, p5+1):
                          for il_2 in range(0, p6+1):
                              coef_v1   = lcoeffs_v1[il_1,il_2]
                              coef_v2   = lcoeffs_v2[il_1,il_2]
                              bi_0      = basis_ad1[ie1, ie2, il_1, 0, g1, g2] * basis_ad2[ie1, ie2, il_2, 0, g1, g2]
                              # ...
                              sx        +=  coef_v1*bi_0
                              sy        +=  coef_v2*bi_0
                              
                    #.. Test 1
                    #rho  = 1.+ 9./(1.+(10.*sqrt((sx-0.7-0.25*0.)**2+(sy-0.5)**2)*cos(arctan2(sy-0.5,sx-0.7-0.25*0.) -20.*((sx-0.7-0.25*0.)**2+(sy-0.5)**2)))**2)
                    #rho   = 1.+5.*exp(-100.*abs((sx-0.45)**2+(sy-0.4)**2-0.1))+5.*exp(-100.*abs(sx**2+sy**2-0.2))+5.*exp(-100*abs((sx+0.45)**2 +(sy-0.4)**2-0.1))  +7.*exp(-100.*abs(sx**2+(sy+1.25)**2-0.4))
                    #.. Test 2 
                    #rho  = 1./(2.+cos(4.*pi*sqrt((sx-0.5-0.25*0.)**2+(sy-0.5)**2)))
                    rho  = 2.+sin(4.*pi*sqrt((sx-0.6)**2+(sy-0.6)**2)) 
                    #.. Test 3 
                    #rho  =  1.+ 3./(1.+(10.*sqrt(sx**2+sy**2)*cos(arctan2(sy,sx) -20.*(sx**2+sy**2)))**2)
                    #.. Test 4 
                    #rho  = 1. + 5./cosh( 5.*((sx-sqrt(3)/2)**2+(sy-0.5)**2 - (pi/2)**2) )**2 + 5./cosh( 5.*((sx+sqrt(3)/2)**2+(sy-0.5)**2 - (pi/2)**2) )**2
                    # .. butterfly
                    #rho   = 1.+7.*exp(-50.*abs((sx)**2+(sy-0.25*times)**2-0.09))
                    uhxx  = lvalues_u1x[g1,g2]
                    uhyy  = lvalues_u2y[g1,g2]
                    uhxy  = lvalues_u1y[g1,g2]

                    v    += (rho*(uhxx*uhyy-uhxy**2)/Crho-1.)**2 * wvol
                    w    += ((x-x1)**2+(y-x2)**2)/abs(uhxx*uhyy-uhxy**2) * wvol
                    if min_det > uhxx*uhyy-uhxy**2 :
                        min_det = uhxx*uhyy-uhxy**2
                        
            Qual_l2      += v
            displacement += w
    rhs[p4,p3]   = sqrt(Qual_l2)
    rhs[p4,p3+1] = sqrt(displacement)
    rhs[p4,p3+2] = min_det
    # ...
