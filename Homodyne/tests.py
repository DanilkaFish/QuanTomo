import unittest
import numpy as np
from homodyne import HomodyneDetection, get_alpha_rho, get_nHSMatrix_psi


class dist(unittest.TestCase):
    def test_utils(self):
        a1, a2, a3 = 1,3,5
        n1, n2, n3 = 1,10,20
        self.assertAlmostEqual(np.trace(get_alpha_rho(a1,n1)), 1)
        self.assertAlmostEqual(np.trace(get_alpha_rho(a2,n2)), 1)
        self.assertAlmostEqual(np.trace(get_alpha_rho(a3,n3)), 1)
        
    def test_HSMatrix(self):
        n = 10
        dx = 0.001
        x = np.arange(-10, 10, step=dx)
        hx = np.array([d for d in get_nHSMatrix_psi(n, x)])
        for i in range(n):
            self.assertAlmostEqual(np.sum(hx[i]*hx[i])*dx, 1, msg=str(i))
        hxhx = np.einsum("ni,mi->nmi", hx, hx.conjugate())
        rho_x = np.einsum("ij,ijn->n", get_alpha_rho(0.5,n), hxhx)
        self.assertAlmostEqual(np.sum(rho_x)*dx, 1, msg=str(i))

if __name__ == "__main__":
    unittest.main()