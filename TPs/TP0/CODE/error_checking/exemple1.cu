#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"

#define THREADS 4096
#define TAB_SIZE 8192

__global__ void kernel(int *a, int *b, int *c) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < TAB_SIZE) c[tid] = a[tid] + b[tid];
}

int main(int argc, char **argv)
{
    int sz_in_bytes = sizeof(int) * TAB_SIZE;

    int *h_c;
    int res = 0;
    int *d_a, *d_b, *d_c;

    // Allocation on host (malloc)
    h_c = (int *)malloc(sz_in_bytes);

    // Allocation on device (cudaMalloc)
    checkCudaErrors(cudaMalloc((void **)&d_a, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_b, sz_in_bytes));
    checkCudaErrors(cudaMalloc((void **)&d_c, sz_in_bytes));

    checkCudaErrors(cudaMemset(d_a, 1, sz_in_bytes));
    checkCudaErrors(cudaMemset(d_b, 2, sz_in_bytes));

    // Kernel configuration
    dim3 dimBlock(THREADS, 1, 1);
    dim3 dimGrid(TAB_SIZE / THREADS + 1, 1, 1);

    // Kernel launch
    kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c);

    // Retrieving data from device (cudaMemcpy)
    checkCudaErrors(cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost));

    // Freeing on device (cudaFree)
    checkCudaErrors(cudaFree(d_a));
    checkCudaErrors(cudaFree(d_b));
    checkCudaErrors(cudaFree(d_c));

    // computing sum of tab element
    for (int i = 0; i < TAB_SIZE; i++) res += h_c[i];

    // Verifying if
    if (res == 3 * TAB_SIZE) {
        fprintf(stderr, "TEST PASSED !\n");
    }
    else
    {
        fprintf(stderr, "TEST FAILED !\n");
    }

    free(h_c);

    return 0;
}
