// compile with `nvcc -o a.out ex00-empty.cu` to create the executable a.out

template <typename T>
__global__ void matrix_add_kernel(unsigned int m, unsigned int n, 
                                  T const* A, T const* B, T* C)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n) {
        C[i + j * m] = A[i + j * m] + B[i + j * m];
    }
}

unsigned int ceildiv(unsigned int n, unsigned int m) {
    return (n + m - 1) / m;
}

template<typename T>
auto matrix_add(unsigned int m, unsigned int n, T const* A, T const* B, T* C, 
                cudaStream_t stream)
{
    dim3 threads(32, 32);
    dim3 blocks(ceildiv(m, threads.x), ceildiv(n, threads.y));
    matrix_add_kernel<<<blocks, threads, 0, stream>>>(m, n, A, B, C);
    return cudaGetLastError();
}


// The following code can be compiled with any C++ compiler

#include <cuda_runtime.h>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <iostream>

template <typename E>
void throw_error(E err) {
    if (err)
        std::cout << "Error: " << err << std::endl;
}

template<typename T>
void rand_matrix(unsigned int m, unsigned int n, T* A) {
    for (unsigned int i=0; i<m; i++) {
        for (unsigned int j=0; j<n; j++) {
            // generate a random number between -1 and 1
            A[i+j*m] = 2 * ((T) rand()) / RAND_MAX - 1;
        }
    }
}

template<typename T>
void test(unsigned int m, unsigned int n) {

    // Allocate and initialize matrices on CPU host
    std::vector<T> hA(m*n), hB(m*n), hC(m*n);
    rand_matrix(m, n, hA.data());
    rand_matrix(m, n, hB.data());

    // Allocate matrices on the device
    T *dA = nullptr;
    T *dB = nullptr;
    T *dC = nullptr;
    size_t size = m * n * sizeof(T);
    throw_error(cudaMalloc(&dA, size));
    throw_error(cudaMalloc(&dB, size));
    throw_error(cudaMalloc(&dC, size));

    // Initialize the RNG
    srand(time(NULL));

    // Copy the input data from the host to the device
    throw_error(cudaMemcpy(dA, hA.data(), size, cudaMemcpyHostToDevice));
    throw_error(cudaMemcpy(dA, hB.data(), size, cudaMemcpyHostToDevice));

    // Add the matrices, using a null stream
    throw_error(matrix_add(m, n, dA, dB, dC, nullptr));

    // Copy the result to the host
    throw_error(cudaMemcpy(hC.data(), dC, size, cudaMemcpyDeviceToHost));

    // Free the device memory
    throw_error(cudaFree(dA));
    throw_error(cudaFree(dB));
    throw_error(cudaFree(dC));
}


int main() {
    test<float>(4, 4);
    return 0;
}
