import unittest
import numpy as np
from task1 import BeamSplitter, get_pn1n2_from_4kron

from utils import * 

class dist(unittest.TestCase):
    def test_get_dist(self):
        n = 3
        a = 1
        rho = np.diag([i for i in puasson(a)(n)])
        rho_twice = np.diag([i for i in puasson(a/2)(n)])
        bs = BeamSplitter(rho)
        rho_evolve = bs.get_evolved_rho()
        rho_theory = np.kron(rho_twice, rho_twice)

        vac = np.zeros((n,n))
        vac[0][0] = 1
        pn1n2 = get_pn1n2_from_4kron(rho_evolve, n)
        pn1n2_theory = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                pn1n2_theory[i,j] = rho_twice[i,i] * rho_twice[j,j]
                
        self.assertTrue(np.isclose(rho_evolve, rho_theory).all)
        self.assertTrue(np.isclose(pn1n2, pn1n2_theory).all)
    
        print(np.array([[12,3],[1,2],[3,9]]))

if __name__ == "__main__":
    unittest.main()