import numpy as np
from functools import partial

from pyccel.epyccel import epyccel

from .linalg import StencilMatrix
from .linalg import StencilVector
from .spaces import TensorSpace

__all__ = ['assemble_matrix', 'assemble_vector', 'assemble_scalar', 'compile_kernel', 'apply_dirichlet']

#==============================================================================
def assemble_matrix(core, V, fields=None, out=None):
    if out is None:
        out = StencilMatrix(V.vector_space, V.vector_space)

    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [u._data for u in fields]

    core( *args, out._data )

    return out

#==============================================================================
def assemble_vector(core, V, fields=None, value = None, out=None):
    if out is None:
        out = StencilVector(V.vector_space)

    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [x._data for x in fields]

    if not(value is None):
        args += [value]

    core( *args, out._data )

    return out

#==============================================================================
def assemble_scalar(core, V, fields=None, values = None):
    # ...
    args = []
    if isinstance(V, TensorSpace):
        args += list(V.nelements)
        args += list(V.degree)
        args += list(V.spans)
        args += list(V.basis)
        args += list(V.weights)
        args += list(V.points)

    else:
        args = [V.nelements,
                V.degree,
                V.spans,
                V.basis,
                V.weights,
                V.points]
    # ...

    if not(fields is None):
        assert(isinstance(fields, (list, tuple)))

        args += [x._data for x in fields]
    
    if not(values is None):
        args += [x for x in values]
    return core( *args )

#==============================================================================
def compile_kernel(core, arity, pyccel=True):
    assert(arity in [0,1,2])

    if pyccel:
        core = epyccel(core)

    if arity == 2:
        return partial(assemble_matrix, core)

    elif arity == 1:
        return partial(assemble_vector, core)

    elif arity == 0:
        return partial(assemble_scalar, core)


#==============================================================================
def apply_dirichlet(V, x):
    if isinstance(x, StencilMatrix):
        if V.dim == 1:
            n1 = V.nbasis

            # ... resetting bnd dof to 0
            x[0,:] = 0.
            x[n1-1,:] = 0.
            # ...

            # boundary x = 0
            x[0,0] = 1.

            # boundary x = 1
            x[n1-1,0] = 1.

            return x

        elif V.dim == 2:
            n1,n2 = V.nbasis

            # ... resetting bnd dof to 0
            x[0,:,:,:] = 0.
            x[n1-1,:,:,:] = 0.
            x[:,0,:,:] = 0.
            x[:,n2-1,:,:] = 0.
            # ...

            # boundary x = 0
            x[0,:,0,0] = 1.

            # boundary x = 1
            x[n1-1,:,0,0] = 1.

            # boundary y = 0
            x[:,0,0,0] = 1.

            # boundary y = 1
            x[:,n2-1,0,0] = 1.

            return x

        elif V.dim == 3:
            n1,n2,n3 = V.nbasis

            # ... resetting bnd dof to 0
            x[0,:,:,:,:,:] = 0.
            x[n1-1,:,:,:,:,:] = 0.
            x[:,0,:,:,:,:] = 0.
            x[:,n2-1,:,:,:,:] = 0.
            x[:,:,0,:,:,:] = 0.
            x[:,:,n3-1,:,:,:] = 0.
            # ...

            # boundary x = 0
            x[0,:,:, 0,0,0] = 1.

            # boundary x = 1
            x[n1-1,:,:, 0,0,0] = 1.

            # boundary y = 0
            x[:,0,:, 0,0,0] = 1.

            # boundary y = 1
            x[:,n2-1,:, 0,0,0] = 1.

            # boundary z = 0
            x[:,:,0, 0,0,0] = 1.

            # boundary z = 1
            x[:,:,n3-1, 0,0,0] = 1.

            return x
        else :
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    elif isinstance(x, StencilVector):
        if V.dim == 1:
            n1 = V.nbasis

            x[0] = 0.
            x[n1-1] = 0.

            return x

        elif V.dim == 2:
            n1,n2 = V.nbasis

            x[0,:] = 0.
            x[n1-1,:] = 0.
            x[:,0] = 0.
            x[:,n2-1] = 0.

            return x

        elif V.dim == 3:
            n1, n2, n3 = V.nbasis

            x[0,:,:] = 0.
            x[n1-1,:,:] = 0.

            x[:,0,:] = 0.
            x[:,n2-1,:] = 0.

            x[:,:,0] = 0.
            x[:,:,n3-1] = 0.

            return x

        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    else:
        raise TypeError('Expecting StencilMatrix or StencilVector')


#==============================================================================
def apply_periodic(V, x, periodic = None, update = None):
       
  if update is None :
    if isinstance(x, StencilMatrix):
        x          = x.tosparse()
        x          = x.toarray()
        if V.dim == 1:
            p  = V.degree
            n1 = V.nbasis

            #... eliminate ghost regions
            li            = np.zeros((2*p, n1))
            li[:p,:]      = x[-p:,:]
            li[p:p+p,-p:] = x[-2*p:-p,-p:]

            x    = x[:-p,:-p] 
            for i in range(p):
                x[i,:]     += li[i,:-p]
                x[i,:p]    += li[i,-p:]       
                x[-1-i,:p] += li[2*p-1-i,-p:]

            return x

        elif V.dim == 2:
          x          = x.reshape((V.nbasis[0],V.nbasis[1], V.nbasis[0], V.nbasis[1]) )
          if True in periodic:
            p1,p2  = V.degree
            n1,n2  = V.nbasis
            #... eliminate ghost regions
            
            if periodic[0] == True:
               lix                       = np.zeros((2*p1, n2, n1, n2))
               lix[:p1,:, :, :]          = x[-p1:,:, :,:]
               lix[p1:p1+p1, :, -p1:, :] = x[-2*p1:-p1,:, -p1:, :]
            
               x    = x[:-p1,:,:-p1,:] 
               for i in range(p1):
                  x[i, :, :, :]      += lix[i, :, :-p1, :]
                  x[i, :, :p1, :]    += lix[i, :,-p1:, :]       
                  x[-1-i, :, :p1, :] += lix[2*p1-1-i, :, -p1:, :]
               n1 = n1 - p1
            if periodic[1] == True:
               liy                       = np.zeros((n1, 2*p2, n1, n2))
               liy[:,:p2,:,:]            = x[:,-p2:, :,:]
               liy[:,p2:p2+p2,:,-p2:]    = x[:, -2*p2:-p2, :, -p2:]
 
               x    = x[:,:-p2, :,:-p2] 
               for j in range(p2):
                  x[:, j, :, :]      += liy[:, j, :, :-p2]
                  x[:, j, :, :p2]    += liy[:, j, :, -p2:]       
                  x[:, -1-j, :, :p2] += liy[:, 2*p2-1-j, :, -p2:]
               n2 = n2 - p2
            x          = x.reshape(( n1*n2,n1*n2 ))                
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one dimension')

        elif V.dim == 3:
          x          = x.reshape((V.nbasis[0],V.nbasis[1],V.nbasis[2], V.nbasis[0],V.nbasis[1],V.nbasis[2]) )
          if True in periodic:
            p1,p2,p3   = V.degree
            n1,n2, n3  = V.nbasis
            #... eliminate ghost regions
            if periodic[0]==True :
               li                         = np.zeros((2*p1,n2,n3, n1,n2,n3))
               li[:p1,:,:, :,:,:]         = x[-p1:,:,:, :,:,:]
               li[p1:p1+p1,:,:, -p1:,:,:] = x[-2*p1:-p1,:,:, -p1:,:,:]
            
               x    = x[:-p1,:,:, :-p1,:,:] 
               for i in range(p1):
                  x[i,:,:, :,:,:]      += li[i,:,:, :-p1,:,:]
                  x[i,:,:, :p1,:,:]    += li[i,:,:, -p1:,:,:]       
                  x[-1-i,:,:,:p1,:,:]  += li[2*p1-1-i,:,:, -p1:,:,:]
               n1 = n1 - p1
            if periodic[1]==True :
               #...
               li                            = np.zeros((n1,2*p2,n3, n1,n2,n3))
               li[:,:p2,:, :,:,:]            = x[:,-p2:,:, :,:,:]
               li[:,p2:p2+p2,:, :,-p2:,:]    = x[:,-2*p2:-p2,:, :,-p2:,:]
 
               x    = x[:,:-p2,:, :,:-p2,:] 
               for j in range(p2):
                  x[:,j,:, :,:,:]      += li[:,j,:, :,:-p2,:]
                  x[:,j,:, :,:p2,:]    += li[:,j,:, :,-p2:,:]       
                  x[:,-1-j,:, :,:p2,:] += li[:,2*p2-1-j,:, :,-p2:,:]
               n2 = n2 - p2
            if periodic[2]==True :
               #...
               li                            = np.zeros((n1,n2,2*p3, n1,n2,n3))
               li[:,:,:p3, :,:,:]            = x[:,:,-p3:, :,:,:]
               li[:,:,p3:p3+p3, :,:,-p3:]    = x[:,:,-2*p3:-p3, :,:,-p3:]
 
               x    = x[:,:,:-p3, :,:,:-p3] 
               for k in range(p3):
                  x[:,:,k, :,:,:]      += li[:,:,k, :,:,:-p3]
                  x[:,:,k, :,:,:p3]    += li[:,:,k, :,:,-p3:]       
                  x[:,:,-1-k, :,:,:p3] += li[:,:,2*p3-1-k, :,:,-p3:]
               n3 = n3 - p3                
            x          = x.reshape(( n1*n2*n3, n1*n2*n3 ))                
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')
        else :
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    elif isinstance(x, StencilVector):
        x          = x.toarray()
        x          = x.reshape(V.nbasis)
        if V.dim == 1:
            #... eliminate ghost regions
            p    = V.degree

            a    = np.zeros(x.shape[0])
            a[:] = x[:]

            x  = x[:-p]
            for i in range(p):
                x[i]             += a[-p+i]
            return x

        elif V.dim == 2:
          if periodic == [True, True] :
            #... eliminate ghost regions
            p1,p2  = V.degree

            a      = np.zeros(x.shape)
            a[:,:] = x[:,:]
            
            x      = x[:-p1,:-p2]
            for i in range(p1):
               for j in range(p2):            
                   x[i,j]            += a[i,-p2+j] + a[-p1+i,j] + a[-p1+i,-p2+j]
            for i in range(p1):
                x[i,p2:]             += a[-p1+i,p2:-p2]
            for j in range(p2):
                x[p1:,j]             += a[p1:-p1,-p2+j]
            x      = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]-p2) ))
            return x

          elif periodic == [True, False] :
            #... eliminate ghost regions
            p1,p2  = V.degree

            a      = np.zeros(x.shape)
            a[:,:] = x[:,:]
            
            x      = x[:-p1,:]
            for i in range(p1):
                x[i,:]             += a[-p1+i,:]
            x      = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]) ))
            return x
            
          elif  periodic == [False, True] :
            #... eliminate ghost regions
            p1,p2  = V.degree

            a      = np.zeros(x.shape)
            a[:,:] = x[:,:]
            
            x     = x[:,:-p2]
            for j in range(p2):
                x[:,j]             += a[:,-p2+j]
            x     = x.reshape(( (V.nbasis[0])*(V.nbasis[1]-p2) ))                                            
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')                

        # ... 
        elif V.dim == 3:
          if periodic == [True, True, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:-p2,:-p3]
            for i in range(p1):
              for j in range(p2):
                for k in range(p3):
                    x[i,j,k]             += a[i,j,-p3+k] + a[i,-p2+j,k]  + a[i,-p2+j,-p3+k] + a[-p1+i,j,k]  + a[-p1+i,j,-p3+k] + a[-p1+i,-p2+j,k] + a[-p1+i,-p2+j,-p3+k]
            # ...
            for i in range(p1):
              for j in range(p2):
                    x[i,j,p3:]           += a[i,-p2+j,p3:-p3] + a[-p1+i,j,p3:-p3]  + a[-p1+i,-p2+j,p3:-p3]
            for i in range(p1):
                for k in range(p3):
                    x[i,p2:,k]           += a[i,p2:-p2,-p3+k] + a[-p1+i,p2:-p2,k]  + a[-p1+i,p2:-p2,-p3+k]
            for j in range(p2):
                for k in range(p3):
                    x[p1:,j,k]           += a[p1:-p1,j,-p3+k] + a[p1:-p1,-p2+j,k]  + a[p1:-p1,-p2+j,-p3+k]
            # ...
            for i in range(p1):
                    x[i,p2:,p3:]         += a[-p1+i,p2:-p2,p3:-p3]
            for j in range(p2):
                    x[p1:,j,p3:]         += a[p1:-p1,-p2+j,p3:-p3]
            for k in range(p3):
                    x[p1:,p2:,k]         += a[p1:-p1,p2:-p2,-p3+k]
                                                                                
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]-p2)*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [True, True, False] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:-p2,:]
            for i in range(p1):
              for j in range(p2):
                    x[i,j,:]             += a[i,-p2+j,:] + a[-p1+i,j,:]  + a[-p1+i,-p2+j,:]
            # ...
            for i in range(p1):
                    x[i,p2:,:]           += a[-p1+i,p2:-p2,:]
            for j in range(p2):
                    x[p1:,j,:]           += a[p1:-p1,-p2+j,:]
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1]-p2)*(V.nbasis[2]) ))
            return x                

          elif periodic == [True, False, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:,:-p3]
            for i in range(p1):
                for k in range(p3):
                    x[i,:,k]             += a[i,:,-p3+k] + a[-p1+i,:,k]  + a[-p1+i,:,-p3+k]
            # ...
            for i in range(p1):
                    x[i,:,p3:]           += a[-p1+i,:,p3:-p3]
            for k in range(p3):
                    x[p1:,:,k]           += a[p1:-p1,:,-p3+k]
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1])*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [False, True, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:,:-p2,:-p3]
            for j in range(p2):
                for k in range(p3):
                    x[:,j,k]             += a[:,j,-p3+k] + a[:,-p2+j,k]  + a[:,-p2+j,-p3+k]   
            # ...
            for j in range(p2):
                    x[:,j,p3:]           += a[:,-p2+j,p3:-p3]
            for k in range(p3):
                    x[:,p2:,k]           += a[:,p2:-p2,-p3+k]
                    
            x          = x.reshape(( (V.nbasis[0])*(V.nbasis[1]-p2)*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [False, False, True] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:,:,:-p3]
            for k in range(p3):
                    x[:,:,k]             += a[:,:,-p3+k] 
            x          = x.reshape(( (V.nbasis[0])*(V.nbasis[1])*(V.nbasis[2]-p3) ))
            return x
          elif periodic == [False, True, False] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:,:-p2,:]
            for j in range(p2):
                    x[:,j,:]             +=  a[:,-p2+j,:]                    

            x          = x.reshape(( (V.nbasis[0])*(V.nbasis[1]-p2)*(V.nbasis[2]) ))
            return x
          elif periodic == [True, False, False] :
            #... eliminate ghost regions
            p1, p2, p3 = V.degree

            a          = np.zeros(x.shape)
            a[:,:,:]   = x[:,:,:]
            
            x          = x[:-p1,:,:]
            for i in range(p1):
                    x[i,:,:]             +=  a[-p1+i,:,:]
                    
            x          = x.reshape(( (V.nbasis[0]-p1)*(V.nbasis[1])*(V.nbasis[2]) ))
            return x
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')                
                             
        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')

    else:
        raise TypeError('ERROR 1 ! Expecting StencilMatrix or StencilVector')

  
  else :
        if V.dim == 1:
            #... update the eliminated ghost regions
            p       = V.degree
            n1      = x.shape[0] + p
            
            a       = np.zeros(n1)
            for i in range(p):
                a[-p-i] = x[i]
            return a

        elif V.dim == 2:
          if periodic == [True, True] :
            #... update the eliminated ghost regions
            p1,p2   = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] + p2
                        
            a       = np.zeros((n1, n2) )           
            a[:-p1,:-p2]  = x[:,:]
            for i in range(p1):
               for j in range(p2):            
                   a[i,-p2+j]     = x[i,j]
                   a[-p1+i,j]     = x[i,j] 
                   a[-p1+i,-p2+j] = x[i,j]

            for i in range(p1):
                a[-p1+i,p2:-p2] = x[i,p2:]

            for j in range(p2):
                a[p1:-p1,-p2+j] = x[p1:,j]                  

            return a

          elif periodic == [True, False] :
            #... update the eliminated ghost regions
            p1,p2      = V.degree
            n1         = x.shape[0] + p1
            n2         = x.shape[1]
                        
            a          = np.zeros((n1, n2))
            a[:-p1,:]  = x[:,:]
            for i in range(p1):
                a[-p1+i,:]   = x[i,:]

            return a

          elif  periodic == [False, True] :
            #... update the eliminated ghost regions
            p1,p2      = V.degree
            n1         = x.shape[0]
            n2         = x.shape[1] + p2
                        
            a          = np.zeros((n1, n2))
            a[:,:-p2]  = x[:,:]
            for j in range(p2):
                a[:,-p2+j]   = x[:,j]                                            
            return a
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')      

        # ... 
        elif V.dim == 3:
          if periodic == [True, True, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] + p2
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:-p2,:-p3]  = x[:,:,:]            
            for i in range(p1):
              for j in range(p2):
                for k in range(p3):
                    a[i,j,-p3+k]          = x[i,j,k]
                    a[i,-p2+j,k]          = x[i,j,k]
                    a[i,-p2+j,-p3+k]      = x[i,j,k]
                    a[-p1+i,j,k]          = x[i,j,k]
                    a[-p1+i,j,-p3+k]      = x[i,j,k]
                    a[-p1+i,-p2+j,k]      = x[i,j,k]
                    a[-p1+i,-p2+j,-p3+k]  = x[i,j,k]
            # ...
            for i in range(p1):
              for j in range(p2):
                    a[i,-p2+j,p3:-p3]     = x[i,j,p3:]
                    a[-p1+i,j,p3:-p3]     = x[i,j,p3:]
                    a[-p1+i,-p2+j,p3:-p3] = x[i,j,p3:]
            for i in range(p1):
                for k in range(p3):
                    a[i,p2:-p2,-p3+k]     = x[i,p2:,k]
                    a[-p1+i,p2:-p2,k]     = x[i,p2:,k]
                    a[-p1+i,p2:-p2,-p3+k] = x[i,p2:,k]
            for j in range(p2):
                for k in range(p3):
                    a[p1:-p1,j,-p3+k]     = x[p1:,j,k]
                    a[p1:-p1,-p2+j,k]     = x[p1:,j,k]
                    a[p1:-p1,-p2+j,-p3+k] = x[p1:,j,k]
            # ...
            for i in range(p1):
                    a[-p1+i,p2:-p2,p3:-p3] = x[i,p2:,p3:]
            for j in range(p2):
                    a[p1:-p1,-p2+j,p3:-p3] = x[p1:,j,p3:]
            for k in range(p3):
                    a[p1:-p1,p2:-p2,-p3+k] = x[p1:,p2:,k]
            return a

          elif periodic == [True, True, False] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] + p2
            n3      = x.shape[2]
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:-p2,:]  = x[:,:,:]            
            for i in range(p1):
              for j in range(p2):
                    a[i,-p2+i,:]     = x[i,j,:]
                    a[-p1+i,j,:]     = x[i,j,:]
                    a[-p1+i,-p2+j,:] = x[i,j,:]
            # ...
            for i in range(p1):
                    a[-p1+i,p2:-p2,:] = x[i,p2:,:]
            for j in range(p2):
                    a[p1:-p1,-p2+j,:] = x[p1:,j,:]
            return a

          elif periodic == [True, False, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] 
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:,:-p3]  = x[:,:,:]            
            for i in range(p1):
                for k in range(p3):
                    a[i,:,-p3+k]      = x[i,:,k]
                    a[-p1+i,:,k]      = x[i,:,k]
                    a[-p1+i,:,-p3+k]  = x[i,:,k]
            # ...
            for i in range(p1):
                    a[-p1+i,:,p3:-p3] = x[i,:,p3:]
            for k in range(p3):
                    a[p1:-p1,:,-p3+k] = x[p1:,:,k]
            return a 

          elif periodic == [False, True, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0]
            n2      = x.shape[1] + p2
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:,:-p2,:-p3]  = x[:,:,:]            
            for j in range(p2):
                for k in range(p3):
                    a[:,j,-p3+k]      = x[:,j,k]
                    a[:,-p2+j,k]      = x[:,j,k]
                    a[:,-p2+j,-p3+k]  = x[:,j,k]         
            # ...
            for j in range(p2):
                    a[:,-p2+j,p3:-p3] = x[:,j,p3:]
            for k in range(p3):
                    a[:,p2:-p2,-p3+k] = x[:,p2:,k]
            return a
            
          elif periodic == [False, False, True] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] 
            n2      = x.shape[1]
            n3      = x.shape[2] + p3
                                    
            a       = np.zeros((n1, n2, n3))
            a[:,:,:-p3]  = x[:,:,:]            
            for k in range(p3):
                    a[:,:,-p3+k] = x[:,:,k]
            return a

          elif periodic == [False, True, False] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] 
            n2      = x.shape[1] + p2
            n3      = x.shape[2]
                                    
            a       = np.zeros((n1, n2, n3))
            a[:,:-p2,:]  = x[:,:,:]            
            for j in range(p2):
                    a[:,-p2+j,:] = x[:,j,:]
            return a
            
          elif periodic == [True, False, False] :
            #... update the eliminated ghost regions
            p1, p2, p3 = V.degree
            n1      = x.shape[0] + p1
            n2      = x.shape[1] 
            n3      = x.shape[2]
                                    
            a       = np.zeros((n1, n2, n3))
            a[:-p1,:,:]  = x[:,:,:]            
            for i in range(p1):
                    a[-p1+i,:,:] = x[i,:,:]
            return a
          else:
             raise NotImplementedError('Only if there is a periodic boundary at least in one direction')                

        else:
            raise NotImplementedError('Only 1d, 2d and 3d are available')
