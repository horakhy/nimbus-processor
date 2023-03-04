import numpy as np
import time
from memory_profiler import memory_usage
from mainAPS import cg_normal_error

# Define a larger linear system
n = 5000
A = np.random.rand(n, n)
b = np.random.rand(n)

# Define the true solution
x_true = np.linalg.solve(A, b)

# Measure the memory consumption
mem_usage = memory_usage((cg_normal_error, (A, b, np.zeros(n))))

# Measure the time taken
try:
    t_start = time.perf_counter()
    x = cg_normal_error(A, b, np.zeros(n))
    t_end = time.perf_counter()
    time_taken = t_end - t_start
except Exception as e:
    print(f"An error occurred: {e}")
    time_taken = None

# Check the solution and residual norm
# if time_taken is not None:
#     assert np.allclose(x, x_true, rtol=1e-6, atol=1e-6)
#     assert np.linalg.norm(b - A.dot(x)) < 1e-6

# Print the results
print(f"Time taken: {time_taken:.3f} seconds")
print(f"Memory usage: {max(mem_usage)} MiB")