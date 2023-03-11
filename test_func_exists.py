import numpy as np
from scipy.sparse import random
from numpy.linalg import norm
from time import time
import ctypes


def test_conjugate_gradient_normal_residual_exists():
    # generate a random sparse Hessian matrix and gradient vector

    myLib = ctypes.CDLL("./mainAPS.cpython-38-x86_64-linux-gnu.so")
    functions = {}
    for func in dir(myLib):
        if callable(getattr(myLib, func)):
            functions[func] = getattr(myLib, func)

    # Print the function names
    for name in functions:
        print(name)
    

test_conjugate_gradient_normal_residual_exists()
