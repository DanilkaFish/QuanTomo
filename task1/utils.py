import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import cm

SIZE = 100

def plot_pn1n2(pn1n2,name="gauss.."):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    n = pn1n2.shape[0]
    T, X = np.meshgrid(np.arange(n), np.arange(n))
    surf = ax.plot_surface(X, T, pn1n2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig(name + '.png', dpi=400)
    
def puasson_dist(n, a=5):
    i = 0 
    p = np.exp(-abs(a)**2)
    yield p
    for i in range(1,n):
        p = p * a*a / i
        yield p

def puasson(a):
    def _puasson(n):
        return puasson_dist(n, a)
    return _puasson  
        
def thermal_state(n, mu=0.5):
    c = np.exp(-mu)
    p = 1-c
    yield p
    for _ in range(1,n):
        p = p * c
        yield p

def ts(mu):
    def _ts(n):
        return thermal_state(n, mu)
    return _ts

def old_g2(sample_data: np.array, bs):
    n = bs.num_basis
    l = len(sample_data)
    n1 = sample_data // n
    n2 = sample_data % n
    n1n2 = n1*n2
    n1_aver = np.sum(n1)
    d2n1 = np.sum(n1*n1)
    n2_aver = np.sum(n2)
    d2n2 = np.sum(n1*n1)
    n1n2_aver = np.sum(n1n2)
    d2n1n2 = np.sum(n1n2*n1n2)
    # print(np.sum())
    d2n1 = (d2n1*l - n1_aver**2)/l
    d2n2 = (d2n2*l - n2_aver**2)/l
    d2n1n2 = (d2n1n2*l - n1n2_aver**2)/l
    
    g = n1n2_aver/n1_aver/n2_aver*l

    d = g*np.sqrt(d2n1/n1_aver**2 + \
                  d2n2/n2_aver**2 + \
                  d2n1n2/n1n2_aver**2)
    return g, d

def g2(sample_data: np.array, bs):
    def get_g2(n1,n2,n1n2):
        return np.sum(n1n2)/np.sum(n1)/np.sum(n2)*SIZE
    n = bs.num_basis
    l = len(sample_data)
    n1 = sample_data // n
    n2 = sample_data % n
    n1n2 = n1*n2
    shots = l//SIZE
    g_array = np.zeros(shots)
    for i in range(shots):
        g_array[i] = get_g2(n1[i * SIZE:(i + 1) * SIZE], 
                            n2[i * SIZE:(i + 1) * SIZE], 
                            n1n2[i * SIZE:(i + 1) * SIZE])
    g = np.sum(g_array)/shots
    return g, shots/(shots - 1) * (np.sum(g_array*g_array)/shots - g**2)