#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


void kernel(double *a, double *b, double *c, int N, int dimBlock, int blockId, int threadId)
{

  int i, j, k;
 	i = blockId * dimBlock + threadId;

 	if (i < N)
 	{
 		c[i] = a[i] + b[i];
	}
}

int main(int argc, char **argv)
{
    int N = 1000;
    int sz_in_bytes = N*sizeof(double);

    double *h_a, *h_b, *h_c;
    double *d_a, *d_b, *d_c;

    h_a = (double*)malloc(sz_in_bytes);
    h_b = (double*)malloc(sz_in_bytes);
    h_c = (double*)malloc(sz_in_bytes);

    // Initiate values on h_a and h_b
    for(int i = 0 ; i < N ; i++)
    {
	h_a[i] = 1./(1.+i);
	h_b[i] = (i-1.)/(i+1.);
    }

    // 3-arrays allocation on device 
//    cudaMalloc((void**)&d_a, sz_in_bytes);
//    cudaMalloc((void**)&d_b, sz_in_bytes);
//    cudaMalloc((void**)&d_c, sz_in_bytes);
    d_a = (double*)malloc(sz_in_bytes);
    d_b = (double*)malloc(sz_in_bytes);
    d_c = (double*)malloc(sz_in_bytes);


    // copy on device values pointed on host by h_a and h_b
    // (the new values are pointed by d_a et d_b on device)

//    cudaMemcpy(d_a, h_a, sz_in_bytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(d_b, h_b, sz_in_bytes, cudaMemcpyHostToDevice);
	memcpy(d_a, h_a, sz_in_bytes);
	memcpy(d_b, h_b, sz_in_bytes);

    //dim3  dimBlock(64, 1, 1);
    //dim3  dimGrid((N + dimBlock.x - 1)/dimBlock.x, 1, 1);
	int dimBlock = (64);
	int dimGrid = ((N + 64 - 1)/64);

  for(int blockId = 0; blockId < dimGrid; ++blockId)
  {
    for(int tid = 0; tid < dimBlock; ++tid)
    {
      kernel(d_a, d_b, d_c, N, dimBlock, blockId, tid);
    }
  }

    // Result is pointed by d_c on device
    // Copy this result on host (result pointed by h_c on host)
//    cudaMemcpy(h_c, d_c, sz_in_bytes, cudaMemcpyDeviceToHost);
	memcpy(h_c, d_c, sz_in_bytes);

    // freeing on device 
//    cudaFree(d_a);
//    cudaFree(d_b);
//    cudaFree(d_c);
    free(d_a);
    free(d_b);
    free(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
