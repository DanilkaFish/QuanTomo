import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
X = np.array([[0, 1], [1, 0]])
I = np.array([[1, 0], [0, 1]])
Url = (I - 1j*X)/np.sqrt(2)
Uda = (I + 1j*Y)/np.sqrt(2)
Uhv = I
U1 = np.array([Uhv, Uda, Url])
    
def rho_projection(rho):
    eigvl, eigenv = np.linalg.eigh(rho)
    while np.any(eigvl<0):
        eigvl = np.where(eigvl < 0, 0, eigvl) 
        eigvl = eigvl - (np.sum(eigvl) - 1)/np.sum(eigvl > 0)
    return  eigenv @ np.diag(eigvl) \
                   @ eigenv.T.conj()

def fidelity_rho_rho(rho1, rho2):
    evalues, evectors = np.linalg.eig(rho1)
    evalues = np.sqrt(evalues)
    rho_sqrt = evectors @ np.diag(evalues) @ evectors.T.conj()
    mead = rho_sqrt @ rho2 @ rho_sqrt
    evalues, _ = np.linalg.eig(mead)
    return np.sum(np.sqrt(evalues))

def process(data: np.array):
    med = np.median(data)
    quan1 = np.quantile(data, 0.25)
    quan2 = np.quantile(data, 0.75)
    return med, quan1, quan2

class state:
    def __init__(self, 
                 hs_size: int = 1):
            self.hs_size = hs_size
            pure = self.generate_psi()
            self.rho = np.outer(pure, pure.conj())
            
    def generate_psi(self):
        num_basis = self.hs_size << 1
        amp = np.random.uniform(0, 1, num_basis)
        return np.sqrt(amp) / np.sqrt(np.sum(amp)) \
                * np.exp(-1j * np.random.uniform(0, 2 * np.pi, num_basis))

    
class tomo:
    def __init__(self, hs_size=1):
        self.hs_size = hs_size
        self.l = hs_size << 1
        self.P = np.zeros((self.l, self.l, self.l))
        for i in range(self.l):
            self.P[i,i,i] = 1
        self.U = U1
        sh = U1.shape
        ush = self.U.shape
        for i in range(1, self.hs_size):
            ush = [i*j for i,j in zip(sh, ush)]
            self.U = np.einsum("nij,mkl->nmikjl",self.U, U1).reshape(ush)
        
        self.Pak= np.einsum("ali,klp,apj->akij", self.U.conj(), self.P, self.U)
        self.B = self.Pak.reshape((-1, self.l * self.l))
        self.pinvB = np.linalg.pinv(self.B)
    
    def get_probs(self, s):
        if (self.hs_size != s.hs_size):
            raise ValueError("System's size mismatch: tomo " + 
                             str(self.hs_size) + " and state " + str(s.hs_size))
        return np.einsum("ij,akij->ak", s.rho, self.Pak).real
    
    def make_tomo_exp(self, probs: np.array, sample_num: int):
        return np.array([np.random.multinomial(sample_num, prob)/sample_num for prob in probs])

    def get_rho_by_pinv(self, sample_probs):
        return rho_projection((self.pinvB @ sample_probs.reshape(-1)).reshape((self.l, self.l)))

    def get_rho_by_cvx(self, sample_probs):
        X = cp.Variable((self.l, self.l), complex=True)
        sample_probs = sample_probs.reshape(-1)
        objective = cp.Minimize(cp.norm(sample_probs - self.B @ cp.vec(X, order="C"), 2))
        constr = (X == X.H, cp.trace(X) == 1, X >> 0 )
        prob = cp.Problem(objective, constr)
        prob.solve("SCS")
        return X.value



if __name__ == "__main__":
    n = 2
    
    n_exp = 100
    n_samples = [10,100,1000]
    med_pinv = np.empty(len(n_samples))
    quan_pinv_up = np.empty(len(n_samples))
    quan_pinv_low = np.empty(len(n_samples))
    med_cvx = np.empty(len(n_samples))
    quan_cvx_up = np.empty(len(n_samples))
    quan_cvx_low = np.empty(len(n_samples))
    fid_pinv = np.empty(n_exp)
    fid_cvx = np.empty(n_exp)
    
    tm = tomo(n)
    s = state(n)
    probs = tm.get_probs(s)
    for j, n_sample in enumerate(n_samples):
        for i in range(n_exp):
            sample = tm.make_tomo_exp(probs, n_sample)
            fid_pinv[i] = fidelity_rho_rho(tm.get_rho_by_pinv(sample), s.rho).real
            fid_cvx[i] = fidelity_rho_rho(tm.get_rho_by_cvx(sample), s.rho).real
        med_pinv[j], quan_pinv_low[j], quan_pinv_up[j] = process(1 - fid_pinv)
        med_cvx[j], quan_cvx_low[j], quan_cvx_up[j] = process(1 - fid_cvx)
    
    plt.errorbar(n_samples, med_pinv, yerr=[quan_pinv_low, quan_pinv_up], label="pinv")
    plt.errorbar(n_samples, med_cvx, yerr=[quan_cvx_low, quan_cvx_up], label="cvx")
    plt.legend()
    plt.loglog()
    plt.savefig("pinv.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
