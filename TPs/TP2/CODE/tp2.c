#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#define NSTREAMS 4

void verif(float *out, int sz)
{
  float err = 0.;

  for(int i = 0 ; i < sz ; i++)
  {
    err += abs(out[i] - exp( - abs(sin(i * 1.0)) ));
  }

  if (err/sz < 1.e-4)
  {
	  fprintf(stdout, "TEST PASSED (error %3.f < 1.e-4)\n", err/sz);
  }
  else
  {
	  fprintf(stderr, "TEST FAILED (error %3.f > 1.e-4)\n", err/sz);
  }
}

void func(float *out, int size)
{
  for(int i = 0; i < size; ++i)
  {
    out[i] = exp( - abs(out[i]) );
  }
}

int main(int argc, char** argv)
{
  int size = 1024;
  if (argc == 2)
  {
	  size = atoi(argv[1]);
  }

  size *= NSTREAMS;

  float *tab = NULL;
  tab = (float*) malloc(sizeof(float) * size);

  if(tab == NULL)
  {
    fprintf(stderr, "Bad allocation\n");
    return -1;
  }

  for(int i = 0; i < size; ++i)
  {
    tab[i] = sin(i * 1.);
  }

  func(tab, size);

  verif(tab, size);

  free(tab);
  return 0;
}
