import numpy as np
from scipy.sparse import random
from numpy.linalg import norm
from time import time
import ctypes


def test_conjugate_gradient_normal_residual():
    # generate a random sparse Hessian matrix and gradient vector
    n = 5000
    H = np.random.randn(n, n)
    g = np.random.randn(n)

    lib = ctypes.CDLL('./mainAPS.cpython-38-x86_64-linux-gnu.so')

    # Specify the argument and return types for the function
    lib.conjugateGradientNormalResidual.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS'),
    ctypes.c_int,
    ctypes.c_double
]
    lib.conjugateGradientNormalResidual.restype = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')

    # compute the true solution using numpy's solve function
    # x_true = np.linalg.solve(H.arr, -g)

    # time the execution of the conjugate_gradient_normal_residual function
    t0 = time()
    x = lib.conjugateGradientNormalResidual(H, g)
    t1 = time()

    # check that the solution is close to the true solution
    # assert norm(x - x_true) / norm(x_true) < 1e-4

    # print the execution time and memory usage
    print(f"Execution time: {t1-t0:.4f} seconds")

test_conjugate_gradient_normal_residual()
