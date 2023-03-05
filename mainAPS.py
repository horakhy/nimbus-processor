import numpy as np
from numba import jit
from memory_profiler import profile

# @jit(cache=True)
def get_f(f, alpha, p):
    return f + alpha * p

# @jit(cache=True)
def get_r(r, alpha, w):
    return r - alpha * w

# @jit(cache=True)
def get_beta(z, z_prev):
    return np.linalg.norm(z)**2 / np.linalg.norm(z_prev)**2

@jit(cache=True)
def get_p(z, beta, p):
    return z + beta * p

@profile
# @jit(forceobj=True, cache=True)
def conjugate_gradient_normal_residual(H, g, max_iter=1000, tol=1e-4):
    n = H.shape[0]
    f = np.zeros(n)
    r = g
    z = H.T.dot(r)
    p = z

    for i in range(max_iter):
        w = H.dot(p)
        alpha = np.linalg.norm(z)**2 / np.linalg.norm(w)**2

        f = get_f(f, alpha, p)
        r = get_r(r, alpha, w)
        
        z_prev = z
        z = H.T.dot(r)

        beta = get_beta(z, z_prev)
        
        p = get_p(z, beta, p)

        if(i % 100 == 1):
            print("Iteration: ", i, "Residual norm: ", np.linalg.norm(r))

        if np.linalg.norm(r) < tol:
            break

    return f
