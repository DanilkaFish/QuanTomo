from __future__ import annotations
import numpy as np
from MLE import MLE, get_nHSMatrix_psi
import matplotlib.pyplot as plt
from scipy.linalg import expm


class HomodyneDetection:
    def __init__(self, rho_meas, alpha_ref, n=None):
        if n is None:
            self.num_basis = rho_meas.shape[0]
        else:
            self.num_basis = n
        self.alpha_ref = alpha_ref
        self.pn1n2 = None
        self.u = None
        self.rho_meas = rho_meas
        self.rho = np.kron(get_alpha_rho(alpha_ref, num_basis), rho_meas)
        self.a = np.diag(np.sqrt(np.arange(1, self.num_basis)), k=1)
        
    def update_rho_ref(self, alpha_ref):
        self.pn1n2 = None
        self.alpha_ref = alpha_ref
        self.rho = np.kron(get_alpha_rho(alpha_ref, num_basis), self.rho_meas)
        
    def get_u(self):
        # U_BS = exp( pi/4 (a+b - b+a))
        # prod = np.pi/4*(np.kron(self.a, self.a.T) - \
        #        np.kron(self.a.T, self.a))
        prod = 1j*np.pi/4*(np.kron(self.a, self.a.T) - \
                np.kron(self.a.T, self.a))
        eigen = np.linalg.eigh(prod)
        u = eigen.eigenvectors @ np.diag(np.exp(-1j*eigen.eigenvalues)) @ \
                                        eigen.eigenvectors.T.conjugate()
        return u
        # return expm(-1j*prod)

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
        p = self.pn1n2.reshape(-1)
        data = np.random.multinomial(n_exp*n_samples, p).reshape(self.pn1n2.shape)
        n = np.empty(2*self.num_basis-1)
        for i in range(-self.num_basis + 1, self.num_basis):
            n[i + self.num_basis - 1] = np.sum(np.diag(data, i))

        return n
    
    
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

        
def meas(rho_meas, num_basis):
    n_exp = 1
    n_samples = 10000
    alpha_ref = 5
    x = np.arange(-num_basis + 1, num_basis)
    
    theta = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    hd = HomodyneDetection(rho_meas, alpha_ref, num_basis)
    exp = np.empty((len(theta), num_basis*2 - 1))
    for i in range(len(theta)):
        hd.update_rho_ref(alpha_ref * np.exp(-1j*theta[i]))
        exp[i] = hd.make_exp(n_exp, n_samples)
        
    x = x/np.sqrt(2)/abs(hd.alpha_ref)
    return exp, x, theta 


if __name__ == "__main__":
    num_basis = 50
    s = np.zeros((num_basis))
    s[1] = 1/np.sqrt(2)
    s[2] = 1/np.sqrt(2)
    rho_fock_12 = np.outer(s,s.conjugate())
    
    n_exp, x, theta = meas(rho_fock_12, num_basis)
    c = MLE(n_exp, x, theta, num_basis)   

    print(c@c.conjugate())
    print(1 - abs(c@s)**2)

    PSI = np.array([arr for arr in get_nHSMatrix_psi(num_basis, x, theta=0)])
    p_theory = np.einsum("ni,nm,mi->i", PSI, rho_fock_12, PSI.conjugate())
    p_mle = np.einsum("n,m,nx,mx->x", c, c.conjugate(), PSI, PSI.conjugate())
    dx = (x[1] - x[0])
    plt.bar(x, n_exp[0]/np.sum(n_exp[0])/dx, width=dx/5)
    plt.plot(x, p_mle.real, label="MLE")
    plt.plot(x, p_theory.real, label="theory")
    plt.legend()
    plt.savefig("homodyne.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()    