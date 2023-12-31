#include <stdio.h>
#include <stdlib.h>
#include "helper_cuda.h"

#define THREADS 64
#define TAB_SIZE 1000

__global__ void copy(int *a, int *b) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < TAB_SIZE) b[tid] = a[tid];
}

__global__ void init(int *a, int value) {
	const int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < TAB_SIZE) a[tid] = value;
}


int main(int argc, char **argv)
{
	int sz_in_bytes = sizeof(int) * TAB_SIZE;

	int *h_b;
	int res = 0;
	int *d_a, *d_b;

	// Allocation on host (malloc)
	h_b = (int *)malloc(sz_in_bytes);

	// Allocation on device (cudaMalloc)
	checkCudaErrors(cudaMalloc((void **)&d_a, sz_in_bytes));
	// suppose we forget this
	// checkCudaErrors(cudaMalloc((void **)&d_b, sz_in_bytes));

	// Kernel configuration
	dim3 dimBlock(THREADS, 1, 1);
	dim3 dimGrid(TAB_SIZE / THREADS + 1, 1, 1);

	init<<<dimGrid, dimBlock>>>(d_a, 1);

	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Kernel launch
	copy<<<dimGrid, dimBlock>>>(d_a, d_b);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	// Retrieving data from device (cudaMemcpy)
	checkCudaErrors(cudaMemcpy(h_b, d_b, sz_in_bytes, cudaMemcpyDeviceToHost));

	// Freeing on device (cudaFree)
	checkCudaErrors(cudaFree(d_a));
	checkCudaErrors(cudaFree(d_b));

	// computing sum of tab element
	for (int i = 0; i < TAB_SIZE; i++) res += h_b[i];

	// Verifying if
	if (res == TAB_SIZE) {
		fprintf(stderr, "TEST PASSED !\n");
	}
	else
	{
		fprintf(stderr, "TEST FAILED !\n");
	}

	free(h_b);

	return 0;
}
