import numpy as np
from math import factorial
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_pn1n2(pn1n2,name="gauss.."):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    n = pn1n2.shape[0]
    # Make data.
    T, X = np.meshgrid(np.arange(n), np.arange(n))
    # Plot the surface.
    surf = ax.plot_surface(X, T, pn1n2, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
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

def g2(n1n2: np.array, bs):
    n = bs.num_basis
    xy, yx = np.meshgrid(np.arange(n), np.arange(n))
    n1_aver = 0
    d2n1 = 0
    n2_aver = 0
    d2n2 = 0
    n1n2_aver = 0
    d2n1n2 = 0
    for i in n1n2:
        n1_aver += xy[i//n, i%n]
        d2n1 += xy[i//n, i%n]**2
        n2_aver += yx[i//n, i%n]
        d2n2 += yx[i//n, i%n]**2
        n1n2_aver += yx[i//n, i%n] * xy[i//n, i%n]
        d2n1n2 += xy[i//n, i%n]**2*yx[i//n, i%n]**2
    d2n1 = (d2n1*len(n1n2) - n1_aver**2)/len(n1n2)
    d2n2 = (d2n2*len(n1n2) - n2_aver**2)/len(n1n2)
    d2n1n2 = (d2n1n2*len(n1n2) - n1n2_aver**2)/len(n1n2)
    
    g = n1n2_aver/n1_aver/n2_aver*len(n1n2)
    
    d = g*np.sqrt(d2n1/n1_aver**2 + \
                  d2n2/n2_aver**2 + \
                  d2n1n2/n1n2_aver**2)
    return n1n2_aver/n1_aver/n2_aver*len(n1n2), d