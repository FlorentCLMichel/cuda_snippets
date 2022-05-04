#include <stdio.h>

// number of threads per block
constexpr unsigned int N_THREADS = 256; 

// compute x ^ m % q 
__device__ unsigned int power(unsigned int m, unsigned int q, unsigned int x);

// n: number of elements
// m: power
// q: modulus
__global__ void powers (unsigned int n, unsigned int m, unsigned int q, unsigned int* x, unsigned int* y);

int main(void) {
    unsigned int n = 1000;
    unsigned int *x, *y, *d_x, *d_y;

    // allocate memory on the host
    x = (unsigned int*) malloc(n * sizeof(unsigned int));
    y = (unsigned int*) malloc(n * sizeof(unsigned int));
    
    // allocate memory on the device
    cudaMalloc(&d_x, n * sizeof(unsigned int));
    cudaMalloc(&d_y, n * sizeof(unsigned int));

    // initialize the host arrays
    for (unsigned int i = 0; i < n; i++) {
        x[i] = (unsigned int) i;
        y[i] = 0.0f;
    }

    // copy the arrays to the device
    cudaMemcpy(d_x, x, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // perform the computation
    unsigned int m = 1000;
    unsigned int q = 27;
    powers<<<(n+N_THREADS-1)/N_THREADS, N_THREADS>>>(n, m, q, d_x, d_y);

    // return the result to the host
    cudaMemcpy(y, d_y, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // print the result
    for (unsigned int i=0; i<n; i++) 
        printf("%d ^ %d %% %d = %d\n", x[i], m, q, y[i]);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    return 0;
}

__device__ unsigned int power (unsigned int m, unsigned int q, unsigned int x) {
    unsigned int y = 1;
    for (unsigned int j = 0; j < m; j++) {
        y *= x;
        y %= q;
    }
    return y;
}

__global__ void powers (unsigned int n, unsigned int m, unsigned int q, 
                        unsigned int* x, unsigned int* y) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = gridDim.x * blockDim.x;
    for (unsigned int i = index; i < n; i+= stride) {
        y[i] = power(m, q, x[i]);
    }
}
