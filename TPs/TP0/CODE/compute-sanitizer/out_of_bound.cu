#include "helper_cuda.h"

__global__ void k1(char *d) { d[41 + 128] = 0; }

int main() {
    char *d;
    checkCudaErrors(cudaMalloc(&d, 42));

    k1<<<1, 1>>>(d);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

}
