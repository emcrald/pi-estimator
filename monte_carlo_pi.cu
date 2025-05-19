#include <curand_kernel.h>

extern "C" __global__ void monte_carlo_pi(int *results, unsigned int seed, int num_points) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= num_points) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    float x = curand_uniform(&state);
    float y = curand_uniform(&state);

    results[idx] = (x * x + y * y <= 1.0f) ? 1 : 0;
}
