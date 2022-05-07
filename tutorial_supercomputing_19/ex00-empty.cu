/*
   An empty Cuda example. 
   Compile with
    * `nvcc -c -o ex00-empty.o ex00-empty.cu` (possibly with additional flags, e.g. `-O3`) to create a .o object
    * `nvcc -o ex00-empty ex00-empty.cu` (possibly with additional flags, e.g. `-O3`) to create an executable
   
    Flags for the Host compiler can be passed using `-Xcompiler "[flags]"`
*/

// GPU kernel (executed by each thread on the GPU)
__global__ void kernel( ... args ... ) {
    ... code ...
}

// CPU driver (executed on CPU)
// The stream is optional
void driver( ... args ..., cudaStream_t stream) {
    dim3 blocks(10, 20, 30); // 10 × 20 × 30 blocks of
    dim3 threads(8, 16);     // 8 × 16 threads each
    size_t shmem = 0;        // size of the shared memory

    // launch a kernel asynchronously and return immediately
    kernel<<<blocks, threads, shmem, stream>>>( ... args ... );
    
    // get and throw the last error
    cudaError_t err = cudaLastError();
    thow_error(err);
}
