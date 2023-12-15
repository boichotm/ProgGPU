#ifndef ELLMATRIX_H
#define ELLMATRIX_H
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include "CSRMatrix.h"

typedef struct
{
  int m_nrows, m_nnz ;
  int m_row_width ;
  double* m_values ;
  int* m_cols ;
} EllMatrix_t;

void init_Ell(EllMatrix_t* A, int nrows, int row_width);
void convert_from_CSR(CSRMatrix_t* A, EllMatrix_t* B);
void mult_Ell(EllMatrix_t* A, double const* x, double* y);

#endif
