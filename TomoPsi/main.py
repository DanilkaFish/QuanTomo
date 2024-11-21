import numpy as np
import matplotlib.pyplot as plt

Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
X = np.array([[0, 1], [1, 0]])
I = np.array([[1, 0], [0, 1]])

def get_psi():
    num_basis = 2
    amp = np.random.uniform(0,1,num_basis)
    return np.sqrt(amp)/np.sqrt(np.sum(amp)) * np.exp(-1j*np.random.uniform(0,2*np.pi, num_basis))

def get_rho():
    theta = np.random.uniform(0, np.pi)
    phi = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(0, 1)
    return 1/2*I + 1/2*r*(X * np.sin(theta) * np.cos(phi) 
                          + Y * np.sin(theta) * np.sin(phi) 
                          + Z * np.cos(theta)
                          ) 
    
def rho_normalization(rho):
    eigvl, eigenv = np.linalg.eigh(rho)
    while np.any(eigvl<0):
        eigvl = np.where(eigvl < 0, 0, eigvl) 
        eigvl = eigvl - (np.sum(eigvl) - 1)/np.sum(eigvl > 0)
    return  eigenv @ np.diag(eigvl) \
                   @ eigenv.T.conj()

class tomo:
    def __init__(self):
        self.P = np.array([[1,0], [0,0]])
        self.P2 = np.array([[0,0],[0,1]])
        ur = (I - 1j*X)/np.sqrt(2)
        ud = (I + 1j*Y)/np.sqrt(2)
        uh = I
        self.U = np.array([uh, ud, ur])
        self.P_aij = np.einsum("ali,lk,akj->aij", self.U.conj(), self.P, self.U)
        self.P2_aij = np.einsum("ali,lk,akj->aij", self.U.conj(), self.P2, self.U)
        
        self.invB = np.linalg.pinv(np.array([*self.P_aij.reshape((3,4)), 
                                             *self.P2_aij.reshape((3,4)) ]))
        
    def get_probs_from_psi(self, psi):
        probs = np.einsum("j,tij,i->t", psi.conj(), self.P_aij, psi)
        return probs 

    def get_probs_from_rho(self, rho):
        probs = np.einsum("ij,tij->t", rho, self.P_aij)
        return probs 
    
    def tomo_rho_1(self, sample_probs):
        p = 2*sample_probs - 1
        return rho_normalization(0.5*np.array([[1 + p[0], p[1] + 1j*p[2]], [p[1] - 1j*p[2], 1 - p[0]]]))
        
    def tomo_rho_2(self, sample_probs):
        probs = np.array([*sample_probs, *(1 - sample_probs)])
        return rho_normalization((self.invB @ probs).reshape((2,2)))
    
def sample_data(probs, sample_num):
    return np.random.binomial(sample_num, probs)/sample_num

def fidelity_rho_rho(rho1, rho2):
    evalues, evectors = np.linalg.eig(rho1)
    evalues = np.sqrt(evalues)
    rho_sqrt = evectors @ np.diag(evalues) @ evectors.T.conj()
    mead = rho_sqrt @ rho2 @ rho_sqrt
    evalues, _ = np.linalg.eig(mead)
    return np.sum(np.sqrt(evalues))

def pure_tomo():
    sample_num = 100
    s = get_psi()
    tm = tomo()
    probs = tm.get_probs_from_psi(s)
    exp_probs = sample_data(probs.real, sample_num)
    rho_1 = tm.tomo_rho_1(exp_probs)
    rho_2 = tm.tomo_rho_2(exp_probs)

    return s.conj() @ rho_1 @ s, s.conj() @ rho_2 @ s, fidelity_rho_rho(rho_1, rho_2)

def mixed_tomo():
    sample_num = 100
    rho = get_rho()
    tm = tomo()
    probs = tm.get_probs_from_rho(rho)
    exp_probs = sample_data(probs.real, sample_num)
    rho_1 = tm.tomo_rho_1(exp_probs)
    rho_2 = tm.tomo_rho_2(exp_probs)

    return fidelity_rho_rho(rho, rho_1), fidelity_rho_rho(rho, rho_2), fidelity_rho_rho(rho_1, rho_2)

if __name__ == "__main__":
    # mixed_tomo()
    n_exp = 100
    fid_1 = np.zeros(n_exp)
    fid_2 = np.zeros(n_exp)
    fid_3 = np.zeros(n_exp)
    for i in range(n_exp):
        fid_1[i], fid_2[i], fid_3[i] = pure_tomo()
    plt.hist(fid_1)
    plt.savefig("fid_1.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.hist(fid_2)
    plt.savefig("fid_2.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.hist(fid_3)
    plt.savefig("fid_3.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()