#include <stdio.h>

// number of threads per block
constexpr unsigned int N_THREADS = 256; 

__global__ void saxpy (unsigned int n, float a, float *x, float *y) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a * x[i];
}

int main(void) {
    unsigned int N = 1<<20;
    float *x, *y, *d_x, *d_y;

    // allocate memory on the host
    x = (float*) malloc(N*sizeof(float));
    y = (float*) malloc(N*sizeof(float));
    
    // allocate memory on the device
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    // initialize the host arrays
    for (unsigned int i = 0; i < N; i++) {
        x[i] = (float) i;
        y[i] = 0.0f;
    }

    // copy the arrays to the device
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // perform the SAXPY operations
    float a = 2.0f;
    saxpy<<<(N+N_THREADS-1)/N_THREADS, N_THREADS>>>(N, a, d_x, d_y);

    // return the result to the host
    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    // compute the maximum error
    float maxError = 0.0f;
    for (unsigned int i=0; i<N; i++) 
        maxError = max(maxError, abs(y[i]-a*x[i]));
    printf("max error: %f\n", maxError);

    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);
    return 0;
}
