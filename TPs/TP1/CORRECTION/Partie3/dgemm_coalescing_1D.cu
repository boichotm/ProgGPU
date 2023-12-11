#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>

#include <cuda.h>
#include <cuda_runtime.h>
#define BLOCK_WIDTH 32
#define TAILLE 4096

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)
/** return time in second
*/
double get_elapsedtime(void)
{
  struct timespec st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

int verify_matrix(double *matRef, double *matOut, int N) {
  double diff = 0.0;
  int i;
  for (i = 0; i < N*N; i++) {
    diff = fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
          matRef[i], matOut[i], diff, i);
      return 1;
    }
  }
  return 0;
}

void init(double* A, double* B, double* C, int size) {
    int i = 0, j = 0;

    srand(2019);

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            A[i * size + j] = (double)(rand() % 10) + 0.01 * (rand() % 5);
            B[i * size + j] = (double)(rand() % 10) + 0.01 * (rand() % 5);
            C[i * size + j] = 0.0;
        }
    }
}

void mult(double* A, double* B, double* C, int size)
{
  int i = 0, j = 0, k = 0;

	for(i = 0; i < size; i++)
  {
		for(j = 0; j < size; j++)
    {
      double sum = 0.;
			for(k = 0; k < size; k++)
      {
			  sum += A[i * size + k] * B[k * size + j];
			}
      C[i * size + j] = sum;
		}
	}
}

// QUESTION 4
__global__
void MulMatrixKernel(double* A, double* B, double* C, int N)
{
  // QUESTION 6
	int line	= threadIdx.x / BLOCK_WIDTH + BLOCK_WIDTH * blockIdx.x;
	int col		= threadIdx.y % BLOCK_WIDTH + BLOCK_WIDTH * blockIdx.y;
  // FIN QUESTION 6

  // QUESTION 7
	if((col < N) && (line < N))
  {
		double val = 0.0f;
		for(int k = 0; k < N; k++)
    {
			val += A[line * N + k] * B[k * N + col];
		}
		C[line * N + col] = val;
	}
  // FIN QUESTION 7
}
// FIN QUESTION 4

int main(int argc, char** argv){
	int N;
  int use_cpu;

	double *A;
	double *B;
	double *C;
	double *C_ref;

  double t0 = 0., t1 = 0., duration = 0.;

	N = (argc < 2)?1000:atoi(argv[1]);
	use_cpu = (argc < 3)?0:atoi(argv[2]);
	fprintf(stdout, "Matrix Multiplication\n  Size: %dx%d\n", N, N);

	// Memory allocation
	A= (double*) malloc(sizeof(double) * N * N);
	B= (double*) malloc(sizeof(double) * N * N);
	C= (double*) malloc(sizeof(double) * N * N);
  if (use_cpu)
  	C_ref = (double*) malloc(sizeof(double) * N * N);

  // timers
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// Value initialization
  init(A, B, C, N);

  // QUESTION 1
	double *d_A, *d_B, *d_C;
	cudaMalloc(&d_A, sizeof(double) * N * N);
	cudaMalloc(&d_B, sizeof(double) * N * N);
	cudaMalloc(&d_C, sizeof(double) * N * N);
  // FIN QUESTION 1

  // QUESTION 2
	cudaMemcpy(d_A, A, sizeof(double) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, sizeof(double) * N * N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, C, sizeof(double) * N * N, cudaMemcpyHostToDevice);
  // FIN QUESTION 2

  // QUESTION 3
	int nbBlocks = N / BLOCK_WIDTH;
	if(N % BLOCK_WIDTH) nbBlocks++;
	dim3 gridSize(nbBlocks, nbBlocks);
	dim3 blockSize(BLOCK_WIDTH * BLOCK_WIDTH);
  // FIN QUESTION 3

  // QUESTION 4
  cudaEventRecord(start);
	MulMatrixKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
  cudaEventRecord(stop);
  // FIN QUESTION 4

  // QUESTION 5
	cudaMemcpy(C, d_C, sizeof(double) * N * N, cudaMemcpyDeviceToHost);
  // FIN QUESTION 5

	// Compute multiplication
  if (use_cpu){
    t0 = get_elapsedtime();
    mult(A, B, C_ref, N);
    t1 = get_elapsedtime();
  }

  // get gpu elapsed time
  cudaEventSynchronize(stop);
  float gpu_duration = 0;
  cudaEventElapsedTime(&gpu_duration, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Pretty print
  uint64_t N_64 = (uint64_t) N;
  uint64_t nb_op = 2* N_64 * N_64 * N_64;

  if (use_cpu){
    duration = (t1 - t0);
    fprintf(stdout, "Performance results: \n");
	  fprintf(stdout, "  Time: %lf s\n", duration);
    fprintf(stdout, "  MFlops: %.2f\n", (nb_op / duration)*1E-6);

    if (verify_matrix(C, C_ref, N) != 0){
      fprintf(stderr, "Wrong result !\n");
    }
  }
  fprintf(stdout, "Performance results (GPU): \n");
  fprintf(stdout, "  Time: %lf s\n", gpu_duration*1E-3);
  fprintf(stdout, "  GFlops: %.2f\n", (nb_op / gpu_duration)*1E-6);


  free(A);
  free(B);
  free(C);

  cudaFree(A);
  cudaFree(B);
  cudaFree(C);

	return 0;
}
