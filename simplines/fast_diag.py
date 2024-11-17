import numpy         as np
from scipy.linalg    import eigh
from scipy.sparse    import csr_matrix, coo_matrix

from .               import fast_diag_core as core
from scipy.sparse    import kron
from scipy.sparse import csr_matrix
from pyccel.epyccel  import epyccel

# =========================================================================
class Poisson(object):
    def __init__(self, mats_1, mats_2, mats_3=None, tau=0.):
        # ...
        assert(len(mats_1) == 2)
        assert(len(mats_2) == 2)

        rdim = None
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
        ds = []
        Us = []
        t_Us = []
        for M, K in zip(Ms, Ks):
            M = M.toarray()
            K = K.toarray()

            d, U = eigh(K, b=M)

            t_U = U.T
            # trick to avoid F/C ordering with pyccel
            U = csr_matrix(U).toarray()
            t_U = csr_matrix(t_U).toarray()


            ds.append(d)
            Us.append(U)
            t_Us.append(t_U)
        # ...

        # ...
        forward  = None
        backward = None
        if rdim == 2:
            U1, U2 = Us[:]
            t_U1, t_U2 = t_Us[:]

            forward  = kron(csr_matrix(t_U1), csr_matrix(t_U2))
            backward = kron(csr_matrix(U1), csr_matrix(U2))

        elif rdim == 3:
            U1, U2, U3 = Us[:]
            t_U1, t_U2, t_U3 = t_Us[:]

            forward  = kron(csr_matrix(t_U1), csr_matrix(t_U2), csr_matrix(t_U3))
            backward = kron(csr_matrix(U1), csr_matrix(U2), csr_matrix(U3))
        # ...

        # ...
        self._mats_1 = mats_1
        self._mats_2 = mats_2
        self._mats_3 = mats_3

        self._ds = ds
        self._Us = Us
        self._t_Us = t_Us
        self._rdim = rdim
        self._tau = tau

        self._forward  = forward
        self._backward = backward
        # ...

    @property
    def rdim(self):
        return self._rdim

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
    def t_Us(self):
        return self._t_Us

    @property
    def tau(self):
        return self._tau

    @property
    def forward(self):
        return self._forward

    @property
    def backward(self):
        return self._backward

    def _solve_2d(self, b):
        # ...
        s_tilde = np.zeros(len(b))
        # ...
        r_tilde = self.forward @ b
        # ...
        core.solve_unit_sylvester_system_2d(*self.ds, r_tilde, float(self.tau), s_tilde)
        s = self.backward @ s_tilde
        return s

    def _solve_3d(self, b):
        # ...
        s_tilde = np.zeros(len(b))
        # ...
        r_tilde = self.forward @ b
        core.solve_unit_sylvester_system_3d(*self.ds, r_tilde, float(self.tau), s_tilde)
        s = self.backward @ s_tilde
        # ...

        return s

    def solve(self, b):
        if self.rdim == 2:
            return self._solve_2d(b)
        else:
            return self._solve_3d(b)
            
            
            
            
            
            
            
            
            
