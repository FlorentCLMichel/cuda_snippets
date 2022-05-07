#include <stdio.h>
constexpr unsigned int N_THREADS = 256;

/// GPU kernel: compute column sums:
///  sums[j] = sum_{i = 0..m} A[i,j]
/// Each block computes one entry.
/// Each thread computes a partial sum and the results are combined.
///
/// Warning: The block size needs to be a power of 2
template <typename T>
__global__ void sum_columns(unsigned int m, const T* A, T* sums)
{
    // partial column sums
    __shared__ T partial_sums[N_THREADS];

    // shift to the jth column
    unsigned int j = blockIdx.x;
    const T* Aj = &A[j*m];

    // compute the partial sum
    unsigned int i = threadIdx.x;
    T temp = 0;
    while (i < m) {
        temp += Aj[i];
        i += blockDim.x;
    }
    partial_sums[threadIdx.x] = temp;

    // synchronize the threads
    __syncthreads();

    // sum reduction
    i = N_THREADS / 2;
    while (i > 0) {
        if (threadIdx.x < i) {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x+i];
        }
        i /= 2;
        __syncthreads();
    }

    // store the result
    if (threadIdx.x == 0) 
        sums[j] = partial_sums[0];
}

/// get the sum of all the elements in the matrix
///
/// Arguments: 
///  * m: number of rows
///  * n: number of columns
///  * A: pointer to host memory containing m*n elements
///  * d_A: pointer to device memory with room for m*n elements
///  * sums: pointer to host memory with room for n elements
///  * d_sums: pointer to device memory with room for n elements
template <typename T>
T sum_matrix(unsigned int m, unsigned int n, const T* A, T* d_A, T* sums, T* d_sums) 
{
    // copy the input to the device
    cudaMemcpy(d_A, A, n*m*sizeof(T), cudaMemcpyHostToDevice);

    // compute the sum of the columns
    sum_columns<<<n, N_THREADS>>> (m, d_A, d_sums);

    // copy the results to the host memory
    cudaMemcpy(sums, d_sums, n*sizeof(T), cudaMemcpyDeviceToHost);

    // sum the results
    T res = 0;
    for (unsigned int i=0; i<n; i++)
        res += sums[i];

    return res;
}

int main (void) {
    
    // matrix size
    unsigned int m = 10000;
    unsigned int n = 20;

    // allocate the host memory
    float* A = (float*) malloc(n*m*sizeof(float));
    float* sums = (float*) malloc(n*sizeof(float));

    // allocate the device memory
    float *d_A, *d_sums; 
    cudaMalloc(&d_A, n*m*sizeof(float));
    cudaMalloc(&d_sums, n*sizeof(float));

    // fill the input
    for (unsigned int i=0; i<m; i++) 
        for (unsigned int j=0; j<n; j++) 
            A[j*m+i] = 1.;

    // compute the sum
    float sum = sum_matrix(m, n, A, d_A, sums, d_sums);
    printf("Result: %f\n", sum);

    // free the device memory
    cudaFree(d_A);
    cudaFree(d_sums);

    // free host memory
    free(A);
    free(sums);

    return 0;
}
