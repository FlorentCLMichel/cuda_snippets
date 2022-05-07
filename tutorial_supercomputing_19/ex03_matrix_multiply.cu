#include <cstdio> // TO REMOVE

// Each thread block will have shape (BLOCK_SIZE, BLOCK_SIZE)
constexpr unsigned int BLOCK_SIZE = 16;

// Matrices are stored in column-major order
// M(row, col) = *(M.elements + row + col*M.stride)
typedef struct {
    unsigned int cols;
    unsigned int rows;
    unsigned int stride;
    float* elements;
} Matrix;

// Get a matrix element
__device__ float GetElement(const Matrix A, unsigned int row, unsigned int col)
{
    return A.elements[row + col*A.stride];
}

// Set a matrix element value
__device__ __host__
void SetElement(Matrix A, unsigned int row, unsigned int col, float value)
{
    A.elements[row + col*A.stride] = value;
}

// Get a sub-matrix with shape (BLOCK_SIZE, BLOCK_SIZE)
__device__ Matrix GetSubMatrix(const Matrix A, unsigned int blockRow, unsigned int blockCol)
{
    Matrix Asub;
    Asub.cols = BLOCK_SIZE;
    Asub.rows = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = A.elements + blockRow * BLOCK_SIZE + blockCol * A.stride * BLOCK_SIZE;
    return Asub;
}

// Matrix multiplication kernel
__global__
void MatMulKernel(const Matrix A, const Matrix B, Matrix C) 
{
    // Block row and column
    unsigned int blockRow = blockIdx.x;
    unsigned int blockCol = blockIdx.y;

    // Row and column within a block
    unsigned int row = threadIdx.x;
    unsigned int col = threadIdx.y;

    // Each thread block computed one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computed one element of Csub
    float Cvalue = 0;

    // loop over the sub-matrices of A and B required to compute Csub
    for (unsigned int k = 0; k < A.cols / BLOCK_SIZE; k++) {

        // Get the sub-matrices of A and B
        Matrix Asub = GetSubMatrix(A, blockRow, k);
        Matrix Bsub = GetSubMatrix(B, k, blockCol);

        // Shared memory used to store Asub and Bsub
        __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

        // Load the submatrices from global to shared memory
        // Each thread loads one element from each sub-matrix
        sA[col][row] = GetElement(Asub, row, col);
        sB[col][row] = GetElement(Bsub, row, col);

        // Synchronize to ensure the sub-matrices are loaded before starting 
        // the computation
        __syncthreads();

        // Multiply Asub and Bsub together
        for (unsigned int j=0; j<BLOCK_SIZE; j++)
            Cvalue += sA[j][row] * sB[col][j];

        // Synchronize the threads to ensure the computation is done before 
        // loading the two new sub-matrices
        __syncthreads();
    }
    
    // Save the result to global memory
    SetElement(Csub, row, col, Cvalue);
}


// Host code
// WARNING: The matrix dimensions must be multiples of BLOCK_SIZE

#include <cassert>

void MatMul(const Matrix d_A, const Matrix d_B, Matrix d_C)
{
    // Check that the matrix dimensions are compatible
    assert(d_A.rows == d_C.rows);
    assert(d_A.cols == d_B.rows);
    assert(d_B.cols == d_C.cols);

    // Invoke the kernel
    dim3 n_threads (BLOCK_SIZE, BLOCK_SIZE);
    dim3 n_blocks (d_C.rows / BLOCK_SIZE, d_C.cols / BLOCK_SIZE);
    MatMulKernel<<<n_blocks, n_threads>>>(d_A, d_B, d_C); 

    auto cudaError = cudaGetLastError();
    if (cudaError) {
        throw cudaError;
    }
}


// A test

#include <vector>
#include <iostream>

int main(void) {

    // matrix shapes
    unsigned int n_rows_A = 1024;
    unsigned int n_cols_A = 2048;
    unsigned int n_rows_B = 2048;
    unsigned int n_cols_B = 512;
    unsigned int n_rows_C = 1024;
    unsigned int n_cols_C = 512;

    // vectors containing the matrix data
    std::vector<float> A(n_rows_A * n_cols_A), B(n_rows_B * n_cols_B), C(n_rows_C * n_cols_C);
    for (unsigned int i=0; i<A.size(); i++)
        A[i] = 2;
    for (unsigned int i=0; i<B.size(); i++)
        B[i] = 0.1;

    // Define the three matrices 
    Matrix d_A {
        .cols = n_cols_A,
        .rows = n_rows_A,
        .stride = n_rows_A,
    };
    Matrix d_B {
        .cols = n_cols_B,
        .rows = n_rows_B,
        .stride = n_rows_B,
    };
    Matrix d_C {
        .cols = n_cols_C,
        .rows = n_rows_C,
        .stride = n_rows_C,
    };
    
    // Allocate device memory
    cudaMalloc(&d_A.elements, n_rows_A*n_cols_A*sizeof(float));
    cudaMalloc(&d_B.elements, n_rows_B*n_cols_B*sizeof(float));
    cudaMalloc(&d_C.elements, n_rows_C*n_cols_C*sizeof(float));

    // Copy the data for A and B to the device memory
    cudaMemcpy(d_A.elements, A.data(), n_rows_A*n_cols_A*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, B.data(), n_rows_B*n_cols_B*sizeof(float), cudaMemcpyHostToDevice);

    // Perform the matrix multiplication
    MatMul(d_A, d_B, d_C);

    // Copy the result to the host memory
    cudaMemcpy(C.data(), d_C.elements, n_rows_C*n_cols_C*sizeof(float), cudaMemcpyDeviceToHost);

    // Check the result
    float error = 0;
    for (unsigned int i=0; i<C.size(); i++)
        error += fabs(C[i] - n_cols_A*0.2);
    float rel_error = error / (C.size()*n_cols_A*0.2);
    std::cout << "Relative error: " << rel_error << std::endl;

    // Free the device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return 0;
}
