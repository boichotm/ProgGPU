#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

int main(int argc, char** argv)
{
  double alpha = 2;
  double *X = NULL, *Y = NULL;
  int N = 1000;
  if (argc > 1) N = atoi(argv[1]);

  X = (double*) malloc(sizeof(double) * N);
  Y = (double*) malloc(sizeof(double) * N);

  for(int i = 0; i < N; ++i)
  {
    X[i] = i;
    Y[i] = X[i] + i;
  }

  for(int i = 0; i < N; ++i)
  {
    Y[i] += alpha * X[i];
  }

  int stop = (N > 5)?5:N;
  for(int i = 0; i < stop; ++i)
    fprintf(stdout, "Y[%d] = %f\n", i, Y[i]);

  return 0;
}
