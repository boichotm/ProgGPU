#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <inttypes.h>
#include <math.h>

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
  uint64_t i;
  uint64_t size = N*N;
  for (i = 0; i < size; i++) {
    diff = fabs(matRef[i] - matOut[i]);
    if (diff > 0.01) {
      printf("Divergence! Should %5.2f, Is %5.2f (Diff %5.2f) at %d\n",
             matRef[i], matOut[i], diff, i);
      return 1;
    }
  }
  return 0;
}


void init(double* A, double* B, double* C, int size)
{
  int i = 0, j = 0;

  srand(2019);

  for(i = 0; i < size; i++)
  {
    for(j = 0; j < size; j++)
    {
      A[i * size + j] = (double) (rand() % 10) + 0.01 * (rand() % 5);
      B[i * size + j] = (double) (rand() % 10) + 0.01 * (rand() % 5);
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

int main(int argc, char** argv){
  int N = 0;

  double *A = NULL;
  double *B = NULL;
  double *C = NULL;

  double t0 = 0., t1 = 0., duration = 0.;

  N = (argc < 2)?1000:atoi(argv[1]);
  fprintf(stdout, "Matrix Multiplication\n  Size: %dx%d\n", N, N);

  // Memory allocation
  A = (double*) malloc(sizeof(double) * N * N);
  B = (double*) malloc(sizeof(double) * N * N);
  C = (double*) malloc(sizeof(double) * N * N);

  // Value initialization
  init(A, B, C, N);

  // Compute multiplication
  t0 = get_elapsedtime();
  mult(A, B, C, N);
  t1 = get_elapsedtime();

  // Pretty print
  duration = (t1 - t0);
  uint64_t N_u64 = (uint64_t) N;
  uint64_t nb_op = N_u64 * N_u64 * N_u64;
  fprintf(stdout, "Performance results: \n");
  fprintf(stdout, "  Time: %lf s\n", duration);
  fprintf(stdout, "  MFlops: %.2f\n", (nb_op / duration)*1E-6);

  return 0;
}
