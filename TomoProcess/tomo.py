
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

import QGOpt as qgo
import tensorflow as tf

Z = np.array([[1, 0], [0, -1]])
Y = np.array([[0, -1j], [1j, 0]])
X = np.array([[0, 1], [1, 0]])
I = np.array([[1, 0], [0, 1]])
CX = np.array([[1,0,0,0],
               [0,1,0,0],
               [0,0,0,1],
               [0,0,1,0]])
II = np.array([1,0,0,0,
               0,1,0,0,
               0,0,1,0,
               0,0,0,1])

psi_tomo = np.array([[1,0], [0,1], [np.sqrt(2)/2, np.sqrt(2)/2] , [np.sqrt(2)/2, -np.sqrt(2)/2]])
Url = (I - 1j*X)/np.sqrt(2)
Uda = (I + 1j*Y)/np.sqrt(2)
Uhv = I
U1 = np.array([Uhv, Uda, Url])
R1 = np.array([np.outer(psi, psi) for psi in psi_tomo])

def process(data: np.array):
    med = np.median(data)
    quan1 = np.quantile(data, 0.25)
    quan2 = np.quantile(data, 0.75)
    return med, quan1, quan2

class TomoProcess:
    def __init__(self, hs_size=1):
        self.hs_size = hs_size
        self.l = 1 << self.hs_size
        self.P = np.zeros((self.l, self.l, self.l))
        for i in range(self.l):
            self.P[i,i,i] = 1
        self.U = U1
        self.R = R1
        sh = U1.shape
        rsh0 = R1.shape
        rsh = R1.shape
        ush = self.U.shape
        for i in range(1, self.hs_size):
            ush = [i*j for i,j in zip(sh, ush)]
            self.U = np.einsum("nij,mkl->nmikjl",self.U, U1).reshape(ush)
            rsh = [i*j for i,j in zip(rsh0, rsh)]
            self.R = np.einsum("nij,mkl->nmikjl",self.R, R1).reshape(rsh)
        self.R = self.R.reshape((self.R.shape[0], -1)).T
        self.Pak = np.einsum("ali,klp,apj->akij", self.U.conj(), self.P, self.U)
        self.B = np.transpose(self.Pak, (0,1,3,2)).reshape((*self.Pak.shape[0:2], self.l * self.l))
    
    def get_probs(self, G):

        P = np.real(np.einsum("akn,nm,mb->abk", self.B, G, self.R).reshape((-1, self.l)))
        P[P<0] = 0
        return P
    

    def get_G(self, U, r):
        E = U.reshape(r, self.l, self.l)
        return np.einsum("rij,rlk->ilkj", E, E.conj()).reshape(self.l*self.l, self.l*self.l)

    def make_tomo_exp(self, probs: np.array, sample_num: int):
        return np.array([np.random.multinomial(sample_num, prob)/sample_num for prob in probs])

    def L(self,k,p):
        return -np.einsum("ijk,ijk->", k, np.ln(p))
    
    def MLE(self, k, r=1, alpha=0.5, ):
        k = k.reshape(-1)
        k = tf.constant(k)
        m = qgo.manifolds.StiefelManifold()
        A = tf.reshape(m.random((self.l*r, self.l), dtype = tf.complex128), (r, self.l, self.l))
        A = qgo.manifolds.complex_to_real(A)
        A = tf.Variable(A)
        opt = qgo.optimizers.RAdam(m , alpha)
        i = 0
        _A = A.numpy() - 1
        while np.sum(np.abs(A.numpy() - _A)) > 0.00001:
            with tf.GradientTape() as tape:
                Ac = qgo.manifolds.real_to_complex(A)
                E = tf.reshape(Ac, (r, self.l, self.l))
                G = tf.reshape(tf.einsum("rij,rlk->ilkj", E, tf.math.conj(E)), (self.l*self.l, self.l*self.l))
                p = tf.reshape(tf.math.real(tf.einsum("akn,nm,mb->abk", self.B, G, self.R)), -1)
                L = -tf.reduce_mean(tf.einsum("i,i->", k, tf.math.log(p)))
            grad = tape.gradient(L,A)
            
            _A = A.numpy()
            opt.apply_gradients(zip([grad], [A]) )
            i += 1
        return qgo.manifolds.real_to_complex(A).numpy()
    

if __name__ == "__main__":
    n = 2
    U = CX
    tm = TomoProcess(n)
    n_exp = 20
    n_samples = [ 100, 1000,10000,100000]
    med_mle       = np.empty(len(n_samples))
    quan_mle_up   = np.empty(len(n_samples))
    quan_mle_low  = np.empty(len(n_samples))
    
    fid_mle       = np.empty(n_exp)
    
    G_id = tm.get_G(U, 1)
    probs = tm.get_probs(G_id)
    for j, n_sample in enumerate(n_samples):
        for i in range(n_exp):
            sample = tm.make_tomo_exp(probs, n_sample)
            E = tm.MLE(sample)
            F = np.sum(np.abs(np.einsum("rij,ij->r",E,U.conj()))**2)/tm.l/tm.l
            fid_mle[i] = F
        med_mle[j], quan_mle_low[j], quan_mle_up[j] = process(1 - fid_mle)
    plt.errorbar(n_samples, med_mle, yerr=[quan_mle_low, quan_mle_up], label="mle for CX")
    plt.legend()
    plt.loglog()
    plt.grid()
    plt.savefig("infid.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    

