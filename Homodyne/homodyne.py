from __future__ import annotations
import numpy as np
from scipy import special as sp
from scipy.linalg import expm
import matplotlib.pyplot as plt
import time

class HomodyneDetection:
    def __init__(self, rho_ref, rho_meas, n=None):
        if n is None:
            self.num_basis = rho_ref.shape[0]
        else:
            self.num_basis = n
            
        self.pn1n2 = None
        self.u = None
        self.rho = np.kron(rho_ref, rho_meas)
        self.a = np.diag(np.sqrt(np.arange(1, self.num_basis)), k=1)
        
    def get_u(self):
        # U_BS = exp( pi/4 (a+b - b+a))
        ts = time.time()
        prod = 1j*np.pi/4*(np.kron(self.a, self.a.T) - \
               np.kron(self.a.T, self.a))
        # eigen = np.linalg.eigh(prod)
        # u = eigen.eigenvectors @ np.diag(np.exp(-1j*eigen.eigenvalues)) @ \
        #                                 eigen.eigenvectors.T.conjugate()
        u = expm(-1j*prod)
        print(time.time() - ts)
        return u
    
    def get_evolved_rho(self):
        if self.u is None:
            self.u = self.get_u()
        return self.u @ self.rho @ self.u.T.conjugate()

    def get_pn1n2(self):
        n = self.num_basis
        if self.pn1n2 is None:
            rho = self.get_evolved_rho()
            self.pn1n2 = np.diagonal(np.diagonal(rho.reshape(n,n,n,n).real, 
                                                 axis1=0, axis2=2), 
                                     axis1=0, axis2=1)
        self.pn1n2 = np.where(self.pn1n2 < 0, 0, self.pn1n2)
        return self.pn1n2
    
    def make_exp(self, n_exp=1, n_samples=100):
        self.get_pn1n2()
        data = np.random.choice(np.arange(self.num_basis**2), size=n_exp*n_samples,
                                p=self.pn1n2.reshape(-1)).reshape(n_exp, n_samples)
        return data // self.num_basis, data % self.num_basis
    
    
def get_nHSMatrix_psi(n, x, theta=0):
    coef = (1/np.sqrt(np.pi))**0.5
    for i in range(n):
        yield coef*sp.HSMatrix(i, monic=True)(x)*np.exp(-x*x/2)
        coef = coef/((i+1)/2)**0.5*np.exp(-1j*theta)
        
def get_alpha_rho(alpha, num_basis):
    def _state(alpha, num_basis):
        p = np.exp(-alpha*alpha.conjugate()/2)
        yield p
        for i in range(1,num_basis):
            p = p*alpha/np.sqrt(i)
            yield p
            
    s = np.array([p for p in _state(alpha, num_basis)])
    s = s/np.sqrt(np.sum(s*s.conjugate()))
    return np.outer(s,s.conjugate())


def theory(rho_meas, n,x,theta=0):
    hx = np.array([d for d in get_nHSMatrix_psi(n, x, theta)])
    # hxhx = np.einsum("ni,mi->nmi", hx, hx.conjugate())
    # rho_x = np.einsum("ij,ijn->n", rho_meas, hxhx)
    rho_x = np.einsum("ni,nm,mi->i", hx, rho_meas, hx.conjugate())
    return rho_x
        
def meas(rho_meas, num_basis, name=''):
    n_exp = 1
    n_samples = 500000
    alpha_ref = 7
    dx = 0.05
    _dx = 0.3
    x = np.arange(-5, 5 + dx, step=dx)
    _x = np.arange(-5, 5 + _dx, step=_dx)   
    for theta in [0]:
        rho_1 = get_alpha_rho(alpha_ref * np.exp(-1j*theta), num_basis) 
        hd = HomodyneDetection(rho_1, rho_meas, num_basis)
        n1, n2 = hd.make_exp(n_exp, n_samples)
        rho_exp0 = (n2 - n1)/np.sqrt(2)/abs(alpha_ref)
        rho_th0 = theory(rho_meas, num_basis, x, theta=theta)
        plt.plot(x, rho_th0.real)
        plt.hist(rho_exp0[0], bins=_x, density=True)
        plt.savefig(name + str(theta)[0:3] + ".png", dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()

if __name__ == "__main__":
    num_basis = 50
    alpha_meas = 1
    rho_2 = get_alpha_rho(alpha_meas, num_basis)
    # meas(rho_2, num_basis, "coh_")
    rho_fock_2 = np.zeros((num_basis,num_basis))
    rho_fock_2[2,2] = 1
    # meas(rho_fock_2, num_basis, "fock_")
    rho_fock_12 = rho_fock_2
    rho_fock_12[1,1] = 0.5
    rho_fock_12[1,2] = 0.5
    rho_fock_12[2,1] = 0.5
    rho_fock_12[2,2] = 0.5
    # meas(rho_fock_12, num_basis, "two_fock_")
    
    rho_mixed = 0.4*rho_2 + 0.6*rho_fock_12
    ts = time.time()
    meas(rho_mixed, num_basis, "mixed_")
    print(time.time() - ts)