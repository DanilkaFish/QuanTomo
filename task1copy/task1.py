from __future__ import annotations
import numpy as np
from utils import puasson, plot_pn1n2, g2, ts

class BeamSplitter:
    def __init__(self, input_rho1):
        self.num_basis = input_rho1.shape[0]
        self.a = np.fromfunction(lambda i,j: (i - j == -1)*np.sqrt(j), 
                                 (self.num_basis, self.num_basis))
        self.u = self.get_u()

        vac = np.zeros((self.num_basis, self.num_basis))
        vac[0][0] = 1
        self.input_rho = np.kron(vac, input_rho1)
        
    def get_u(self):
        # U = exp( pi/4 (a+ab - b+a+a))
        prod = 1j*np.pi/4*(np.kron(self.a, self.a.T) - \
               np.kron(self.a.T, self.a))
        eigen = np.linalg.eigh(prod)
        return eigen.eigenvectors @ np.diag(np.exp(-1j*eigen.eigenvalues)) @ \
                                        eigen.eigenvectors.T.conjugate()

    def get_evolved_rho(self):
        return self.u @ self.input_rho @ self.u.T.conjugate()

    
        
def get_pn1n2_from_4kron(rho,n):
    rho = rho.reshape(n,n,n,n).real
    return np.diagonal(np.diagonal(rho, axis1=(0), axis2=(2)), axis1=0, axis2=1)

def get_g2(rho, size=100, name="puasson"):
    bs = BeamSplitter(rho)
    pn1n2 = get_pn1n2_from_4kron(bs.get_evolved_rho(), bs.num_basis)
    plot_pn1n2(pn1n2, name)
    ls = np.random.choice(np.arange(bs.num_basis**2), size=size, p=pn1n2.reshape(-1)/np.sum(pn1n2))
    return g2(ls, bs) 

if __name__ == "__main__":
    n = 5
    rho_ps = np.diag([i for i in puasson(2)(5)])

    coh_in_g, coh_in_d = get_g2(rho_ps)
    print("puasson -- g2 = ", coh_in_g, "+-" ,coh_in_d)

    n = 20
    rho_ts = np.diag([i for i in ts(0.2)(n)])
    ts_in_g, ts_in_d = get_g2(rho_ts, name="thermal")
    print("heat_state -- g2 = ", ts_in_g, "+-", ts_in_d)
    
    