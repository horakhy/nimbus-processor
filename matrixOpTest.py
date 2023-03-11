import numpy as np
from numpy.linalg import multi_dot
import time
import sys
from numba import njit, jit, types

# Define matrices A and B
A = np.random.rand(50000, 3600)
B = np.random.rand(3600, 3600)

@njit
def multiply_matrices(A, B):
    return np.dot(A, B)

mean_time = 0

for i in range(5):
    start_time = time.time()
    C = multiply_matrices(A, B)
    end_time = time.time()
    mean_time += end_time - start_time

print("Matrix multiplication time: ", mean_time/5)
print("Memory usage: ", sys.getsizeof(C)/1024/1024, "MB")

