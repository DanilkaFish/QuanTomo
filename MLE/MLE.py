from scipy import special as sp
import numpy as np
import time
import matplotlib.pyplot as plt

def get_nHSMatrix_psi(n, x, theta=0):
    coef = (1/np.sqrt(np.pi))**0.5
    for i in range(n):
        yield coef*sp.HSMatrix(i, monic=True)(x)*np.exp(-x*x/2)
        coef = coef/((i+1)/2)**0.5*np.exp(-1j*theta)
        

def MLE(n_exp, x, theta, num_basis):
    PSI = np.array([arr for arr in get_nHSMatrix_psi(num_basis, x, theta=0)])
    THETA = np.exp(-1j*np.tensordot(theta, np.arange(num_basis), axes=0))
    PSI_THETA = np.einsum("nx,tn->nxt", PSI, THETA)
    lam = np.sum(n_exp)
    c = np.zeros(num_basis)
    c[1] = 1
    c[2] = 1
    c = c / np.sqrt(np.sum(c))
    c_pred = np.zeros(num_basis)
    def R(c):
        # ts = time.time()
        # arr = np.einsum("nxt,mxt,k,k,kxt,kxt->nm", PSI_THETA, PSI_THETA.conjugate(), 1/c, 1/c.conjugate(), 1/PSI_THETA, 1/PSI_THETA.conjugate())
        # print(time.time() - ts)
        # ts = time.time()
        p = np.einsum("m,n,nxt,mxt->tx", c, c.conjugate(), PSI_THETA, PSI_THETA.conjugate())
        arr = np.einsum("tx,nxt,mxt,tx->nm", n_exp, PSI_THETA, PSI_THETA.conjugate(), 1/p)
        return arr
        
    def next(c, alpha):
        c = alpha / lam * R(c) @ c + (1 - alpha)*c
        return c
    
    while (np.max(np.abs(c_pred - c)) > 0.00001):
        c_pred = c 
        c = next(c, 0.8)
    return c

def theory(rho_meas, n,x,theta=0):
    hx = np.array([d for d in get_nHSMatrix_psi(n, x, theta)])
    rho_x = np.einsum("ni,nm,mi->i", hx, rho_meas, hx.conjugate())
    return rho_x.real

if __name__ == "__main__":
    num_basis = 50
    theta = np.array([0, np.pi/4, np.pi/2, np.pi*3/4])
    dx = 0.1
    x = np.arange(-15, 15 + dx, step=dx)
    
    amp = np.random.uniform(0,1,num_basis)
    s = np.sqrt(amp)/np.sqrt(np.sum(amp)) * np.exp(-1j*np.random.uniform(0,2*np.pi))
    rho = np.outer(s, s.conjugate())
    
    n_exp = np.empty((len(theta), len(x))) 
    for i in range(len(theta)):
        n_exp[i] = np.random.multinomial(1000, theory(rho, num_basis, x, theta[i])*dx)
    c = MLE(n_exp, x, theta, num_basis)
    
    print("fideleity: ", 1 - abs(c@s)**2)
    
    PSI = np.array([arr for arr in get_nHSMatrix_psi(num_basis, x, theta=0)])
    p_theory = np.einsum("n,m,nx,mx->x", s, s.conjugate(), PSI, PSI.conjugate())
    p_mle = np.einsum("n,m,nx,mx->x", c, c.conjugate(), PSI, PSI.conjugate())
    plt.plot(x, p_theory.real, label="theory")
    plt.plot(x, p_mle.real, label="MLE")
    plt.savefig("random_state.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    
    