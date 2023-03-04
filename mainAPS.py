import numpy as np

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
        ap_array = A.dot(p)
        alpha = rs_old / np.dot(p, ap_array)

        if np.isinf(alpha) or np.isnan(alpha):
            raise ValueError("CGMN failed: alpha is invalid")

        x += alpha * p
        r -= alpha * ap_array
        rs_new = np.dot(r, r)

        if np.sqrt(rs_new) < tol:
            break

        if rs_new == 0:
            raise ValueError("CGMN failed: rs_new is zero")

        if rs_new < tol**2:
            break

        beta = rs_new / rs_old

        if np.isinf(beta) or np.isnan(beta):
            raise ValueError("CGMN failed: beta is invalid")

        if(i % 100 == 1):
            print("Iteration: ", i, "Residual norm: ", np.sqrt(rs_new))    
        p = r + beta * p
        rs_old = rs_new

    return x