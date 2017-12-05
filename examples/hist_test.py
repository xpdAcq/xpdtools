import numpy as np
from numba import jit, prange

result = np.zeros(2000)
result2 = np.zeros(2000)
a = (2048, 2048)
values = np.random.random(a)
xy = np.random.randint(0, 2000, a)


def run():
    for i in np.unique(xy):
        result2[i] = np.max(values[xy == i])

#@jit(nopython=True, parallel=True)
def inner(result, vfs, h):
    i = 0
    for j in prange(len(h)):
        k = h[j]
        result[j] = np.max(vfs[i:i+k])
        i += k


#@jit(nopython=True, nogil=True)
def run_sort(xy, result, values):
    xf = xy.flatten()
    vf = values.flatten()
    idx = xf.argsort()
    
    vfs = vf[idx]

    h = np.bincount(xf)
    inner(result, vfs, h)

