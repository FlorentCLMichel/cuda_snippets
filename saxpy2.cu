#include <iostream>
#include <math.h>

constexpr unsigned int BLOCKSIZE = 256;

// kernel function
__global__ void saxpy (unsigned int n, float a, float* x, float* y) {
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride)
        y[i] += a * x[i];
}

int main (void) {

    unsigned int N = 1<<20;
    float *x, *y;

    // allocate unified memory, accessible from both the host and device
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize the arrays on the host
    for (unsigned int i = 0; i < N; i++) {
        x[i] = sin((float) i);
        y[i] = 0.;
    }

    // run the SAXPY operations on the device
    float a = 2;
    unsigned int numBlocks = (N + BLOCKSIZE - 1) / BLOCKSIZE;
    saxpy<<<numBlocks, BLOCKSIZE>>>(N, a, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();
    
    // compute the maximum error
    float maxError = 0.0f;
    for (unsigned int i=0; i<N; i++) { 
        maxError = fmax(maxError, fabs(y[i]-a*x[i]));
    }
    std::cout << "max error: " << maxError << std::endl;

    cudaFree(x);
    cudaFree(y);
    return 0;
}
