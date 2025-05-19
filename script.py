import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import time

NUM_POINTS = 500_000_000
BLOCK_SIZE = 1024
GRID_SIZE = (NUM_POINTS + BLOCK_SIZE - 1) // BLOCK_SIZE

mod = cuda.module_from_file("monte_carlo_pi.ptx")
monte_carlo_pi = mod.get_function("monte_carlo_pi")

results_gpu = cuda.mem_alloc(NUM_POINTS * np.int32().nbytes)

start = time.time()
monte_carlo_pi(
    results_gpu,
    np.uint32(np.random.randint(1, 1 << 30)),
    np.int32(NUM_POINTS),
    block=(BLOCK_SIZE, 1, 1),
    grid=(GRID_SIZE, 1)
)

results_host = np.empty(NUM_POINTS, dtype=np.int32)
cuda.memcpy_dtoh(results_host, results_gpu)
end = time.time()

inside_circle = np.sum(results_host)
pi_estimate = 4 * inside_circle / NUM_POINTS

print(f"Estimated π: {pi_estimate}")
print(f"Points: {NUM_POINTS}")
print(f"Time taken: {end - start:.2f} seconds")
