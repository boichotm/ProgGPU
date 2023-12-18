#include <mpi.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define TSIZE 1024




__global__ void addVector(int * tab_a, int * tab_b, int * tab_c)
{
	int index;
	index = blockIdx.x * blockDim.x + threadIdx.x;
	tab_c[index] = tab_a[index] + tab_b[index];
}


int verif_tab(int * tab, int size)
{
	int i, ret = 0;

	for(i=0; i<size; i++)
	{
		if(tab[i] != size)
		{
			 ret=1;
			printf("i=%d ; tab[i]=%d\n", i, tab[i]);
		}
	}

	return ret;
}



int main(int argc, char*argv[])
{
	int rank, size, i;

	int *tab_a;
	int *tab_b;
	int *tab_c;
	int *tab_d;

	int *h_a;
	int *h_b;
	int *h_c;
	int *d_a;
	int *d_b;
	int *d_c;

	MPI_Init(&argc, &argv);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);


	if(rank==0)
	{
		tab_a = (int *)malloc(sizeof(int)*TSIZE*size);
		tab_b = (int *)malloc(sizeof(int)*TSIZE*size);
		tab_c = (int *)malloc(sizeof(int)*TSIZE*size);
		tab_d = (int *)malloc(sizeof(int)*TSIZE*size);

		h_a = (int *)malloc(sizeof(int)*TSIZE);
		h_b = (int *)malloc(sizeof(int)*TSIZE);
		h_c = (int *)malloc(sizeof(int)*TSIZE);

		struct timeval mpi_start;
		struct timeval mpi_stop;
		struct timeval cuda_start;
		struct timeval cuda_stop;


		for(i=0; i<TSIZE*size; i++)
		{
			tab_a[i] = i;
			tab_b[i] = TSIZE*size-i;
			tab_c[i] = 0;
		}
		gettimeofday(&mpi_start, NULL);


		MPI_Scatter(tab_a, TSIZE, MPI_INT, h_a, TSIZE, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(tab_b, TSIZE, MPI_INT, h_b, TSIZE, MPI_INT, 0, MPI_COMM_WORLD);


		for(i=0; i<TSIZE; i++) h_c[i] = h_a[i] + h_b[i];

		MPI_Gather(h_c, TSIZE, MPI_INT, tab_c, TSIZE, MPI_INT, 0, MPI_COMM_WORLD);

		gettimeofday(&mpi_stop, NULL);

		int res;

		res = verif_tab(tab_c, TSIZE*size);

		if(res == 0)
		{
			printf("CPU Compute is OK ; time is %f s\n", (float)(mpi_stop.tv_sec - mpi_start.tv_sec) + ((float)(mpi_stop.tv_usec - mpi_start.tv_usec))/1000000);
		}

		cudaMalloc((void**) &d_a, sizeof(int)*TSIZE*size);
		cudaMalloc((void**) &d_b, sizeof(int)*TSIZE*size);
		cudaMalloc((void**) &d_c, sizeof(int)*TSIZE*size);



		gettimeofday(&cuda_start, NULL);
		cudaMemcpy(d_a, tab_a, sizeof(int)*TSIZE*size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, tab_b, sizeof(int)*TSIZE*size, cudaMemcpyHostToDevice);

		addVector<<<size, TSIZE>>>(d_a, d_b, d_c);

		cudaMemcpy(tab_d, d_c, sizeof(int)*TSIZE*size, cudaMemcpyDeviceToHost);
		gettimeofday(&cuda_stop, NULL);

		res = verif_tab(tab_d, TSIZE*size);

		if(res == 0)
		{
			printf("GPU Compute is OK ; time is %f s\n", (float)(cuda_stop.tv_sec - cuda_start.tv_sec) + ((float)(cuda_stop.tv_usec - cuda_start.tv_usec))/1000000);
		}

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);

	}
	else
	{
		h_a = (int *)malloc(sizeof(int)*TSIZE);
		h_b = (int *)malloc(sizeof(int)*TSIZE);
		h_c = (int *)malloc(sizeof(int)*TSIZE);

		MPI_Scatter(tab_a, TSIZE, MPI_INT, h_a, TSIZE, MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatter(tab_b, TSIZE, MPI_INT, h_b, TSIZE, MPI_INT, 0, MPI_COMM_WORLD);


		for(i=0; i<TSIZE; i++) h_c[i] = h_a[i] + h_b[i];

		MPI_Gather(h_c, TSIZE, MPI_INT, tab_c, TSIZE, MPI_INT, 0, MPI_COMM_WORLD);

	}



	MPI_Finalize();

	return 0;
}
