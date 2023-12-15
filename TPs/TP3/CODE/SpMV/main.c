#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#include "omp.h"

#include "CSRMatrix.h"
#include "EllMatrix.h"

int main(int argc, char** argv)
{
  double *x = NULL, *y = NULL;
  int nx = 100, ny = 100, nrows = 0;
  int nb_test = 1, i = 0, j = 0;
  int check = 1;
  double t0 = 0., t1 = 0., duration = 0.;
  double norme= 0. ;
  CSRMatrix_t* cpu_matrix = NULL;
  EllMatrix_t* ell_matrix = NULL;

  if(argc > 1) nx = atoi(argv[1]);
  if(argc > 2) ny = atoi(argv[2]);
  if(argc > 3) nb_test = atoi(argv[3]);
  if(argc > 4) check = atoi(argv[4]);
  nrows = nx * ny;

  fprintf(stdout, "NX: %d\tNY: %d\tNTest: %d\n", nx, ny, nb_test);
  cpu_matrix = (CSRMatrix_t*) malloc( sizeof(CSRMatrix_t) );
  ell_matrix = (EllMatrix_t*) malloc( sizeof(EllMatrix_t) );
  x = (double*) malloc( nrows * sizeof(double) );
  y = (double*) malloc( nrows * sizeof(double) );
  for(i = 0; i < nrows; ++i)
  {
    x[i] = 1. * i ;
    y[i] = 0. ;
  }

  buildLaplacian(cpu_matrix,nx,ny) ;
  convert_from_CSR(cpu_matrix, ell_matrix);

  #ifdef DEBUG
  print_CSR(cpu_matrix);
  #endif

  for(i = 0; i < nrows; ++i)
    x[i] = 1. * i ;

  mult_Ell(ell_matrix, x, y);

#ifndef SEQ
  int nb_part = 4;
  if(argc > 5) nb_part = atoi(argv[5]);
  int* offset = NULL;
  offset = (int*) malloc(sizeof(int) * (nb_part + 1));
  offset[0] = 0;
  offset[nb_part] = nrows;
  int chunk = nrows / nb_part;

  for(i = 1; i < nb_part; ++i)
  {
    offset[i] = offset[i-1] + chunk;
#ifdef DEBUG
    fprintf(stdout, "%d> from %d to %d\n", i-1, offset[i-1], offset[i]);
#endif //DEBUG
  }
#ifdef DEBUG
  fprintf(stdout, "%d> from %d to %d\n", nb_part-1, offset[nb_part-1], offset[nb_part]);
#endif //DEBUG

#else //SEQ
  fprintf(stdout, "Sequential version...\n");
#endif //SEQ

  for(i = 0; i < nb_test; ++i)
  {
    t0 = get_elapsedtime();

#ifdef SEQ
    mult_CSR(cpu_matrix,x,y) ;
#else
#pragma omp parallel num_threads(nb_part)
  {
#pragma omp single nowait
    {
      for(int k = 0; k < nb_part; ++k)
      {
#pragma omp task shared(cpu_matrix, x, y, offset)
        {
          mult_CSR_task(cpu_matrix, x, y, offset[k], offset[k+1]);
        }
      }
    }
  }
#endif

    t1 = get_elapsedtime();
    duration += (t1 - t0);

    norme=0. ;
    for(j=0;j<nrows;++j)
      norme += y[j]*y[j] ;
    norme = sqrt(norme) ;
    for(j=0;j<nrows;++j)
      x[j] = y[j]/norme ;
  }

  if(check)
  {
    double norme=0. ;
    for(i=0;i<nrows;++i)
       norme += y[i]*y[i] ;
    fprintf(stdout, "NORME Y= %.2f\n",sqrt(norme)) ;
  }

  fprintf(stdout, " Time : %f\n", duration);
  uint64_t flop_csr = (unsigned long long)(cpu_matrix->m_nnz) * 2;
  fprintf(stdout, " MFlops : %.2f\n", flop_csr / (duration/nb_test)*1E-6);
  fprintf(stdout, "AvgTime : %f\n", duration/nb_test);

  free(x);
  free(y);
  destruct_CSR(cpu_matrix);
  free(cpu_matrix);

  return 0;
}
