import time
import numpy as np
from scipy.special import gamma
def _state(alpha, num_basis):
    p = np.exp(-alpha*alpha.conjugate()/2)
    yield p
    for i in range(1,num_basis):
        p = p*alpha/np.sqrt(i)
        yield p
        
alpha = 5.4
n = 10000
ts = time.time()
s = np.array([p for p in _state(alpha, n)])
print(time.time() - ts)


ts = time.time()
s = np.exp(-alpha*alpha.conjugate()/2)*gamma(np.arange(n))
    # s = np.array([p for p in _state(alpha, n)])
print(time.time() - ts)