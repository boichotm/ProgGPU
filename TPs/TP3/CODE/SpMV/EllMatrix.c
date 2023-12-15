#include "EllMatrix.h"

void init_Ell(EllMatrix_t* A, int nrows, int row_width)
{
  A->m_values = (double*) calloc(nrows * row_width, sizeof(double));
  A->m_cols = (int*) calloc(nrows * row_width, sizeof(int));
  A->m_nrows = nrows;
  A->m_nnz = nrows * row_width;
  A->m_row_width = row_width;
}

void convert_from_CSR(CSRMatrix_t* A, EllMatrix_t* B)
{
  B->m_nrows = A->m_nrows;
  B->m_nnz = A->m_nnz;
  int max_row_width = -1;
  for(int i = 0; i < B->m_nrows; ++i)
  {
    int tmp = A->m_kcol[i+1] - A->m_kcol[i];
    if(tmp > max_row_width)
    {
      max_row_width = tmp;
    }
  }
  B->m_row_width = max_row_width;

#ifdef DEBUG
  fprintf(stdout, "Max row width: %d\n", max_row_width);
  fprintf(stdout, "Matrix Size: %d\n", max_row_width * B->m_nrows);
#endif

  B->m_values = (double*) calloc(B->m_nrows * max_row_width, sizeof(double));
  B->m_cols   = (int*)    calloc(B->m_nrows * max_row_width, sizeof(int));

  for(int i = 0; i < A->m_nrows; ++i)
  {
    for(int k = A->m_kcol[i], j = 0; k < A->m_kcol[i+1]; ++k, ++j)
    {
      B->m_cols[  (i * max_row_width) + j ] = A->m_cols[k];
      B->m_values[(i * max_row_width) + j ] = A->m_values[k];
    }
  }
}

void mult_Ell(EllMatrix_t* A, double const* x, double* y)
{
  const int N = A->m_nrows;
  const int row_width = A->m_row_width;

  /* A COMPLETER */
}
