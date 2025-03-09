"""
fast_diag.py

# fast diagonalization solver for Laplace equation

@author : M. BAHARI
"""

import numpy         as np
from scipy.linalg    import eigh
from scipy.sparse    import csr_matrix, coo_matrix

from .               import fast_diag_core as core
from scipy.sparse    import kron
from scipy.sparse    import csr_matrix
from pyccel.epyccel  import epyccel

# =========================================================================
class Poisson(object):
    def __init__(self, mats_1, mats_2, mats_3=None, tau=0.):
        # ...
        assert(len(mats_1) == 2)
        assert(len(mats_2) == 2)


        rdim  = None
        if mats_3 is None:
            rdim = 2
        else:
            assert(len(mats_3) == 2)
            rdim = 3
        # ...

        # ...
        if rdim == 2:
            Ms = [mats_1[0], mats_2[0]]
            Ks = [mats_1[1], mats_2[1]]
        else:
            Ms = [mats_1[0], mats_2[0], mats_3[0]]
            Ks = [mats_1[1], mats_2[1], mats_3[1]]
        # ...

        # ... generalized eigenvalue decomposition
        nDoFs = []
        ds    = []
        Us    = []
        for M, K in zip(Ms, Ks):
            M = M.toarray()
            K = K.toarray()

            d, U = eigh(K, b=M)

            # trick to avoid F/C ordering with pyccel
            U   = csr_matrix(U).toarray()


            ds.append(d)
            Us.append(U)
            nDoFs.append(len(d))

        # ...
        self._mats_1 = mats_1
        self._mats_2 = mats_2
        self._mats_3 = mats_3

        self._ds    = ds
        self._Us    = Us
        self._rdim  = rdim
        self._tau   = tau
        self._nDoFs = nDoFs
        # ...

    @property
    def rdim(self):
        return self._rdim
    
    @property
    def nDoFs(self):
        return self._nDoFs

    @property
    def mats_1(self):
        return self._mats_1

    @property
    def mats_2(self):
        return self._mats_2

    @property
    def mats_3(self):
        return self._mats_3

    @property
    def ds(self):
        return self._ds

    @property
    def Us(self):
        return self._Us

    @property
    def tau(self):
        return self._tau

    def _solve_2d(self, b):

        # # ... Avoidding kron product
        n1, n2  = self.nDoFs
        s_tilde = b.reshape((n1,n2))
        s_tilde = self.Us[0].T @ s_tilde @ self.Us[1]
        core.solve_unit_sylvester_system_2d(*self.ds, s_tilde, float(self.tau), s_tilde)
        s_tilde = self.Us[0] @ s_tilde @ self.Us[1].T
        s_tilde = s_tilde.reshape(n1*n2)
        return s_tilde

    def _solve_3d(self, b):
        # ... Avoidding kron product TODO optimize and parallelize
        n1, n2, n3 = self.nDoFs
        s_tilde    = b.reshape(n1,n2*n3)
        s_tilde    = s_tilde.T @ self.Us[0]
        # matrix becomes (n2*n3, n1)
        for i1 in range(n1):
            r_tilde         = s_tilde[:,i1]
            # ...
            r_tilde         = r_tilde.reshape((n2, n3))
            r_tilde         = self.Us[1].T @ r_tilde @ self.Us[2]
            #... r_tilde(i2, i3) = r_tilde(i2, i3) / (ds[0](i1,0) + ds[1](i2,0) + ds[2](i3,0) + _tau);
            core.solve_unit_sylvester_system_2d(self.ds[1],self.ds[2], r_tilde, float(self.tau+self.ds[0][i1]), r_tilde)
            r_tilde         = self.Us[1] @ r_tilde @ self.Us[2].T
            s_tilde[:,i1]   = r_tilde.reshape(n2*n3)
        '''
        from joblib import Parallel, delayed

        def process_column(i1):
            r_tilde = s_tilde[:, i1].copy().reshape((n2, n3))
            r_tilde = self.Us[1].T @ r_tilde @ self.Us[2]
            core.solve_unit_sylvester_system_2d(self.ds[1], self.ds[2], r_tilde, float(self.tau + self.ds[0][i1]), r_tilde)
            r_tilde = self.Us[1] @ r_tilde @ self.Us[2].T
            return i1, r_tilde.reshape(n2 * n3)

        # Run in parallel
        results = Parallel(n_jobs=-1)(delayed(process_column)(i1) for i1 in range(n1))
        # Store results
        for i1, r_tilde in results:
            s_tilde[:, i1] = r_tilde'
        '''
        #...
        s_tilde = self.Us[0] @ s_tilde.T
        s_tilde = s_tilde.reshape(n1*n2*n3)
        # ...
        return s_tilde

    def _project_2d(self, b):
        # # ... Avoidding kron product
        n1, n2  = self.nDoFs
        s_tilde = b.reshape((n1,n2))
        s_tilde = self.Us[0].T @ s_tilde @ self.Us[1]
        s_tilde = self.Us[0] @ s_tilde @ self.Us[1].T
        s_tilde = s_tilde.reshape(n1*n2)
        return s_tilde

    def _project_3d(self, b):
        # ... Avoidding kron product
        n1, n2, n3 = self.nDoFs
        s_tilde    = b.reshape(n1,n2*n3)
        s_tilde    = s_tilde.T @ self.Us[0]
        # matrix becomes (n2*n3, n1)
        for i1 in range(n1):
            r_tilde         = s_tilde[:,i1]
            r_tilde         = r_tilde.reshape((n2, n3))
            r_tilde         = self.Us[1].T @ r_tilde @ self.Us[2]
            r_tilde         = self.Us[1] @ r_tilde @ self.Us[2].T
            s_tilde[:,i1]   = r_tilde.reshape(n2*n3)
        s_tilde = self.Us[0] @ s_tilde.T
        s_tilde = s_tilde.reshape(n1*n2*n3)
        # ...
        return s_tilde

    
    def solve(self, b):
        if self.rdim == 2:
            return self._solve_2d(b)
        else:
            return self._solve_3d(b)

    def project(self, b):
        if self.rdim == 2:
            return self._project_2d(b)
        else:
            return self._project_3d(b)
            
            
            
            
            
            
            
            
            
