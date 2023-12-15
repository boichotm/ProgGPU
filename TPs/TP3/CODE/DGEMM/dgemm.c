#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

typedef struct timespec struct_time;
#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9*(double)t.tv_nsec)
/** return time in second
*/
double get_elapsedtime(void)
{
  struct_time st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

void mult(int N, float *A, float* B, float* C)
{
  int i = 0, j = 0, k = 0;
  int total_size = N*N;
  for(i = 0; i < N; ++i)
  {
    for(j = 0; j < N; ++j)
    {
      float val = 0.;
      for(k = 0; k < N; ++k)
      {
        val += A[i * N + k] * B[k * N + j];
      }
      C[i * N + j] = val;
    }
  }
}

int main(int argc, char** argv)
{
  double t0 = 0., t1 = 0., duration = 0.;

  float *A = NULL;
  float *B = NULL;
  float *C = NULL;

  int i = 0;

  int N = 1024;
  if(argc > 1)
  {
    N = atoi(argv[1]);
  }

  fprintf(stdout, "> Matrix Multiplication Kernel...\n");
  fprintf(stdout, "    Size: %dx%d\n", N, N);
#pragma omp parallel default(none) shared(stdout)
  {
#pragma omp single
    fprintf(stdout, "    Running on %d threads\n", omp_get_num_threads());
  }

  A = (float*) calloc(N * N, sizeof(float));
  B = (float*) calloc(N * N, sizeof(float));
  C = (float*) calloc(N * N, sizeof(float));

  for(i = 0; i < N * N; ++i)
  {
    A[i] = 1. * i;
    B[i] = N*N - (1. * i);
    C[i] = 0.;
  }

  t0 = get_elapsedtime();
  mult(N, A, B, C);
  t1 = get_elapsedtime();

  duration = (t1 - t0);

  fprintf(stdout, "    Elapsed Time : %f\n", duration);

  return 0;
}
