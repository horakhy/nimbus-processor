import numpy as np
from numba import jit

@jit(cache=True)
def get_alpha(A, r, p):
    """
    Get the alpha value for the Conjugate Gradient Method.

    Parameters:
    A (ndarray): Coefficient matrix of the linear system.
    r (ndarray): Residual vector.
    p (ndarray): Search direction vector.

    Returns:
    alpha (float): Alpha value.
    """
    alpha = np.dot(r, r) / np.dot(p, A.dot(p))

    # Handle invalid alpha value
    if not np.isfinite(alpha):
        alpha = 0

    return alpha

@jit(cache=True)
def get_beta(rs_new, rs_old):
    return rs_new / rs_old

@jit(cache=True) 
def support_operation(r, beta, p):
    return r + beta * p

@jit(cache=True)
def cg_normal_error(A, b, x0, max_iter=1000, tol=1e-4):
    """
    Conjugate Gradient Method Normal Error algorithm to solve linear system Ax = b.

    Parameters:
    A (ndarray): Coefficient matrix of the linear system.
    b (ndarray): Right-hand side vector of the linear system.
    x0 (ndarray): Initial guess for the solution vector.
    max_iter (int): Maximum number of iterations.
    tol (float): Tolerance level for the residual norm.

    Returns:
    x (ndarray): Solution vector.
    """
    x = np.array(x0)
    r = b - A.dot(x)
    p = r
    # residual = tol
    
    ap_array = np.zeros_like(b)
    rs_old = np.dot(r, r)

    for i in range(max_iter):
        alpha = get_alpha(A, r, p)

        x += alpha * p
        r -= alpha * ap_array
        rs_new = np.dot(r, r)

        # Handle zero residual case
        if rs_new == 0:
            break

        if np.sqrt(rs_new) < tol:
            break

        beta = get_beta(rs_new, rs_old)

        # Handle invalid beta value
        if not np.isfinite(beta):
            beta = 0

        # if(i % 100 == 1):
        #     print("Iteration: ", i, "Residual norm: ", np.sqrt(rs_new))    
        p = support_operation(r, beta, p)

        rs_old = rs_new

    return x