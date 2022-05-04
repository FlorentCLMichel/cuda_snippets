#include <stdio.h>

__global__ void say_hello() {
    printf("Hello from the GPU!\n");
}

int main() {
    printf("Hello from the CPU!\n");
    say_hello<<<1,1>>>();
    cudaDeviceSynchronize();
    return 0;
}
