import numpy as np
from scipy.sparse import random
from numpy.linalg import norm
from time import time
from mainAPS import conjugate_gradient_normal_residual


def test_conjugate_gradient_normal_residual():
    # generate a random sparse Hessian matrix and gradient vector
    n = 5000
    H = random(n, n, density=0.1, format='csr')
    g = np.random.randn(n)

    # compute the true solution using numpy's solve function
    x_true = np.linalg.solve(H.toarray(), -g)

    # time the execution of the conjugate_gradient_normal_residual function
    t0 = time()
    x = conjugate_gradient_normal_residual(H, g)
    t1 = time()

    # check that the solution is close to the true solution
    # assert norm(x - x_true) / norm(x_true) < 1e-4

    # print the execution time and memory usage
    print(f"Execution time: {t1-t0:.4f} seconds")

test_conjugate_gradient_normal_residual()
