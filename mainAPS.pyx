import numpy as np
cimport numpy as np

"# distutils: language=3"

cdef conjugateGradientNormalResidual(np.ndarray[double, ndim=2] H, np.ndarray[double, ndim=1] g, int max_iter=1000, double tol=1e-4):
    cdef int n = H.shape[0]
    cdef np.ndarray[double, ndim=1] f = np.zeros(n)
    cdef np.ndarray[double, ndim=1] r = g.copy()
    cdef np.ndarray[double, ndim=1] z = H.T.dot(r)
    cdef np.ndarray[double, ndim=1] p = z.copy()

    cdef np.ndarray[double, ndim=1] w = np.empty(n)
    cdef double alpha, beta

    for i in range(max_iter):
        w = H.dot(p)
        alpha = np.linalg.norm(z)**2 / np.linalg.norm(w)**2

        f = f + alpha * p
        r = r - alpha * w
        
        z_prev = z.copy()
        z = H.T.dot(r)

        beta = np.linalg.norm(z)**2 / np.linalg.norm(z_prev)**2
        
        p = z + beta * p

        if(i % 100 == 1):
            print("Iteration: ", i, "Residual norm: ", np.linalg.norm(r))

        if np.linalg.norm(r) < tol:
            break

    return f

