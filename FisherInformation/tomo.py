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
        

def rho_projection(rho, get_c=False, r=None):
    eigvl, eigenv = np.linalg.eigh(rho)
    while np.any(eigvl<0):
        eigvl = np.where(eigvl < 0, 0, eigvl) 
        eigvl = eigvl - (np.sum(eigvl) - 1)/np.sum(eigvl > 0)

    if get_c:
        if r is None:
            r = np.sum(eigvl>0)
        return eigenv[:, -r:] @ np.diag(np.sqrt(eigvl[-r:]))
    return  eigenv @ np.diag(eigvl) \
                   @ eigenv.T.conj()

def fidelity_rho_rho(rho1, rho2):
    evalues, evectors = np.linalg.eig(rho1)
    evalues = np.sqrt(evalues)
    rho_sqrt = evectors @ np.diag(evalues) @ evectors.T.conj()
    mead = rho_sqrt @ rho2 @ rho_sqrt
    evalues, _ = np.linalg.eig(mead)
    return np.sum(np.sqrt(evalues))**2

def get_chi(k, probs, n_exp):
    return np.sum((n_exp*k[:,0:-1] - n_exp*probs[:,0:-1])**2/n_exp/probs[:,0:-1]/(1 - probs[:,0:-1]))

def process(data: np.array):
    med = np.median(data)
    quan1 = np.quantile(data, 0.25)
    quan2 = np.quantile(data, 0.75)
    return med, quan1, quan2


class state:
    def __init__(self, 
                 hs_size: int = 1):
        self.hs_size = hs_size
        self.pure = self.generate_psi()
        self.rho = np.outer(self.pure, self.pure.conj())
            
    def generate_psi(self):
        num_basis =  1 << self.hs_size
        amp = np.random.uniform(0, 1, num_basis)
        return np.sqrt(amp) / np.sqrt(np.sum(amp)) \
                * np.exp(-1j * np.random.uniform(0, 2 * np.pi, num_basis))

    
class tomo:
    def __init__(self, hs_size=1):
        self.hs_size = hs_size
        self.l = 1 << self.hs_size
        self.P = np.zeros((self.l, self.l, self.l))
        for i in range(self.l):
            self.P[i,i,i] = 1
        self.U = U1
        sh = U1.shape
        ush = self.U.shape
        for i in range(1, self.hs_size):
            ush = [i*j for i,j in zip(sh, ush)]
            self.U = np.einsum("nij,mkl->nmikjl",self.U, U1).reshape(ush)
        self.Pak = np.einsum("ali,klp,apj->akij", self.U.conj(), self.P, self.U)
        self.B = np.transpose(self.Pak, (0,1,3,2)).reshape((-1, self.l * self.l))
        self.pinvB = np.linalg.pinv(self.B)
    
    def get_probs(self, s):
        if (self.hs_size != s.hs_size):
            raise ValueError("System's size mismatch: tomo " + 
                             str(self.hs_size) + " and state " + str(s.hs_size))
        return np.einsum("ji,akij->ak", s.rho, self.Pak).real
        
    def get_fisher_matrix(self, probs, c, n):
        w = np.einsum("akij,jr->irak",self.Pak, c)
        w = w.reshape((-1, *w.shape[2:]))
        c = c.reshape(-1)
        v = np.array([*np.real(w), *np.imag(w)])
        ctilda = np.array([*np.real(c), *np.imag(c)])
        F = 4*n*np.einsum("iak,jak,ak->ij", v, v, 1/probs)\
            + 4*(n*n - n) * probs.shape[0] * np.einsum("i,j->ij", ctilda, ctilda)
        return F

    def make_tomo_exp(self, probs: np.array, sample_num: int):
        return np.array([np.random.multinomial(sample_num, prob)/sample_num for prob in probs])

    def get_rho_by_pinv(self, sample_probs):
        return (self.pinvB @ sample_probs.reshape(-1)).reshape((self.l, self.l))

    def get_rho_by_cvx(self, sample_probs):
        X = cp.Variable((self.l, self.l), complex=True)
        sample_probs = sample_probs.reshape(-1)
        objective = cp.Minimize(cp.norm(sample_probs - self.B @ cp.vec(X, order="C"), 2))
        constr = (X == X.H, cp.trace(X) == 1, X >> 0 )
        prob = cp.Problem(objective, constr)
        prob.solve("SCS")
        return X.value



    def MLE(self, k, c0, alpha=0.8):
        lam = np.sum(k)

        def J(c):
            pak = np.einsum("akns,nm,sm->ak", self.Pak, c.conj(), c)
            arr = np.einsum("ak,ak,akps->ps", k, 1/pak, self.Pak)
            return arr
            
        def next(c, alpha):
            c = alpha / lam * np.einsum("pq,qr->pr", J(c), c) + (1 - alpha)*c
            return c
        
        c = next(c0, alpha)
        c_pred = c0
        while (np.max(np.abs(c_pred - c)) > 0.0000001):
            c_pred = c 
            c = next(c, 0.8)
        # rho = np.einsum("pq,rq->pr", c, c.conj())
        return c
    

if __name__ == "__main__":
    n = 2
    n_exp = 100
    n_samples = [ 100, 1000,10000,100000]
    
    med_pinv      = np.empty(len(n_samples))
    inf_theory    = np.empty(len(n_samples))
    dinf_theory    = np.empty(len(n_samples))
    med_mle       = np.empty(len(n_samples))
    quan_mle_up   = np.empty(len(n_samples))
    quan_mle_low  = np.empty(len(n_samples))
    
    fid_pinv      = np.empty(n_exp)
    fid_mle       = np.empty(n_exp)
    fid_cvx       = np.empty(n_exp)
    
    chi           = np.empty(n_exp)
    tm = tomo(n)
    s = state(n)
    probs = tm.get_probs(s)
    for j, n_sample in enumerate(n_samples):
        for i in range(n_exp):
            sample = tm.make_tomo_exp(probs, n_sample)
            chi[i] = get_chi(sample, probs, n_sample)
            
            almost_rho = tm.get_rho_by_pinv(sample)
            c0         = rho_projection(almost_rho, get_c=True, r=1)
            c          = tm.MLE(sample, c0=c0)
            rho_mle    = np.einsum("pq,rq->pr", c, c.conj())

            fid_mle[i]  = fidelity_rho_rho(rho_mle, s.rho).real

        c_theory = np.zeros(c.shape, dtype=np.complex128)
        c_theory[:,0] = s.pure
        F = tm.get_fisher_matrix(probs, c_theory, n_sample)
        eigv = np.linalg.eig(F).eigenvalues
        d = np.array([1/h for h in eigv if np.abs(h)>0.00001])
        inf_theory[j] = np.sum(d)
        dinf_theory[j] = np.sqrt(np.sum(np.outer(d,d)) + np.sum(d*d*2) - inf_theory[j]**2)
        med_mle[j], quan_mle_low[j], quan_mle_up[j] = process(1 - fid_mle)
    
    plt.errorbar(n_samples, med_mle, yerr=[quan_mle_low, quan_mle_up], label="mle")
    plt.errorbar(n_samples, inf_theory, yerr=[inf_theory - dinf_theory, inf_theory + dinf_theory], label="via Fisher")
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.savefig("infid.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    
    print(np.quantile(chi, 0.95))
    plt.hist(chi, density=True)
    plt.savefig("chi.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
