#include "CSRMatrix.h"
#include "omp.h"

/** return time in second
*/
double get_elapsedtime(void)
{
  struct_time st;
  int err = gettime(&st);
  if (err !=0) return 0;
  return (double)st.tv_sec + get_sub_seconde(st);
}

void init_CSR(CSRMatrix_t* A, int nrows, int nnz)
{
  A->m_nrows = nrows ;
  A->m_nnz = nnz ;
  A->m_values = (double*) malloc(nnz * sizeof(double) );
  A->m_cols = (int*) malloc(nnz * sizeof(int) );
  A->m_kcol = (int*) malloc((nnz + 1) * sizeof(int) );
}

void destruct_CSR(CSRMatrix_t* A)
{
  free(A->m_values);
  free(A->m_cols);
  free(A->m_kcol);
}

void mult_CSR(CSRMatrix_t* A, double const* x, double* y)
{
  int i = 0, k = 0, nrows = A->m_nrows;
  double val = 0.;
  double* values = A->m_values;
  int* kcol = A->m_kcol;
  int* cols = A->m_cols;

  for(i=0;i<nrows;++i)
  {
    val = 0. ;
    for(k=kcol[i];k<kcol[i+1];++k)
      val += values[k]*x[cols[k]] ;
    y[i] = val ;
  }
}

void mult_CSR_task(CSRMatrix_t* A, double const* x, double* y, int start, int end)
{
  int i = 0, k = 0, nrows = A->m_nrows;
  double val = 0.;
  double* values = A->m_values;
  int* kcol = A->m_kcol;
  int* cols = A->m_cols;

  if( start < 0 || start > nrows || end <0 || end > nrows)
  {
    fprintf(stderr, "FATAL ERROR !\n");
    return;
  }

  for(i = start; i<end; ++i)
  {
    val = 0. ;
    for(k=kcol[i];k<kcol[i+1];++k)
      val += values[k]*x[cols[k]] ;
    y[i] = val ;
  }
}

void print_CSR(CSRMatrix_t* A)
{
  int i = 0, k = 0, nrows = A->m_nrows;
  double* values = A->m_values;
  int* kcol = A->m_kcol;
  int* cols = A->m_cols;
  fprintf(stdout, "NROWS: %d | NNZ: %d\n", nrows, A->m_nnz);
  for(i=0;i<nrows;++i)
  {
    fprintf(stdout, "ROW [%d]\n\t", i);
    for(k=kcol[i];k<kcol[i+1];++k)
      fprintf(stdout, "(%d: %f) ", cols[k], values[k]);
    fprintf(stdout, "\n");
  }
}

int hat(int i,int n)
{
  return max(0,min(i,n-1)) ;
}

int uid(int i,int j,int nx,int ny)
{
  return hat(j,ny)*nx+hat(i,nx) ;
}

double  _trans_m_i(double* perm,int i,int j,int nx,int ny)
{
  double p1 = perm[uid(i-1,j,nx,ny)] ;
  double p2 = perm[uid(i,j,nx,ny)] ;
  return p1*p2/(p1+p2) ;
}

double  _trans_p_i(double* perm,int i,int j,int nx,int ny)
{
  double p1 = perm[uid(i+1,j,nx,ny)] ;
  double p2 = perm[uid(i,j,nx,ny)] ;
  return p1*p2/(p1+p2) ;
}

double  _trans_m_j(double* perm,int i,int j,int nx, int ny)
{
  double p1 = perm[uid(i,j-1,nx,ny)] ;
  double p2 = perm[uid(i,j,nx,ny)] ;
  return p1*p2/(p1+p2) ;
}

double  _trans_p_j(double* perm,int i,int j,int nx, int ny)
{
  double p1 = perm[uid(i,j+1,nx,ny)] ;
  double p2 = perm[uid(i,j,nx,ny)] ;
  return p1*p2/(p1+p2) ;
}

void buildLaplacian(CSRMatrix_t* matrix,
    int nx, int ny)
{

  int i=0,j=0, nrows = nx*ny;
  int nnz = 5*(nx-2)*(ny-2) + (nx+ny-4)*8+4*3 ;
  fprintf(stdout, "NROWS     : %d\n", nrows) ;
  fprintf(stdout, "NNZ       : %d\n\n", nnz) ;
  init_CSR(matrix,nrows,nnz) ;
  double* m_permitivity = NULL ;
  m_permitivity = (double*) malloc(nrows * sizeof(double) );
  for(i = 0; i < nrows; ++i) m_permitivity[i] = 1.;

  int* cols = matrix->m_cols ;
  int* kcol = matrix->m_kcol ;
  double* values = matrix->m_values ;
  int irow =0 ;
  int offset = 0 ;
  {
    j=0 ;
    {
      i=0 ;
      double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
      double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
      double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
      double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

      int row_size = 3 ;
      kcol[irow] = offset ;
      cols[offset] = irow ;
      cols[offset+1] = irow+1 ;
      cols[offset+2] = irow+nx ;
      values[offset] = T_p_i + T_p_j ;
      {
        values[offset] += T_m_i;
      }
      {
        values[offset] += T_m_j;
      }
      values[offset+1] = -T_p_i ;
      values[offset+2] = -T_p_j ;
      offset += row_size ;
      ++irow ;
    }
    for(i=1;i<nx-1;++i)
    {
      double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
      double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
      double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
      double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

      int row_size = 4 ;
      kcol[irow] = offset ;
      cols[offset] = irow-1 ;
      cols[offset+1] = irow ;
      cols[offset+2] = irow+1 ;
      cols[offset+3] = irow+nx ;
      values[offset] = -T_m_i ;
      values[offset+1] = T_m_i+T_p_i+T_p_j ;
      {
        values[offset+1] += T_m_j;
      }
      values[offset+2] = -T_p_i ;
      values[offset+3] = -T_p_j ;
      offset += row_size ;
      ++irow ;
    }
    {
        i=nx-1 ;
        double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
        double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
        double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
        double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

        int row_size = 3 ;
        kcol[irow] = offset ;
        cols[offset] = irow-1 ;
        cols[offset+1] = irow ;
        cols[offset+2] = irow+nx ;
        values[offset] = -T_m_i ;
        values[offset+1] = T_m_i+T_p_j ;
        values[offset+2] = -T_p_j ;
        {
          values[offset+1] += T_p_i;
        }
        {
          values[offset+1] += T_m_j;
        }
        offset += row_size ;
        ++irow ;
     }
  }
  for(j=1;j<ny-1;++j)
  {
    {
        i=0 ;
        double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
        double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
        double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
        double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

        int row_size = 4 ;
        kcol[irow] = offset ;
        cols[offset] = irow-nx ;
        cols[offset+1] = irow ;
        cols[offset+2] = irow+1 ;
        cols[offset+3] = irow+nx ;
        values[offset] = -T_m_j ;
        values[offset+1] = T_m_j+T_p_i+T_p_j ;
        values[offset+2] = -T_p_i ;
        values[offset+3] = -T_p_j ;
        {
          values[offset+1] += T_m_i;
        }
        offset += row_size ;
        ++irow ;
    }
    for(i=1;i<nx-1;++i)
    {
      int row_size = 5 ;
      double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
      double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
      double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
      double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

      kcol[irow] = offset ;
      cols[offset] = irow-nx ;
      cols[offset+1] = irow -1;
      cols[offset+2] = irow ;
      cols[offset+3] = irow+1 ;
      cols[offset+4] = irow+nx ;
      values[offset] = -T_m_j ;
      values[offset+1] = -T_m_i ;
      values[offset+2] = T_m_j+T_m_i+T_p_i+T_p_j ;
      values[offset+3] = -T_p_i ;
      values[offset+4] = -T_p_j ;
      offset += row_size ;
      ++irow ;
    }
    {
        i=nx-1 ;
        double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
        double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
        double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
        double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

        int row_size = 4 ;
        kcol[irow] = offset ;
        cols[offset] = irow-nx ;
        cols[offset+1] = irow-1 ;
        cols[offset+2] = irow ;
        cols[offset+3] = irow+nx ;
        values[offset] = -T_m_j ;
        values[offset+1] = -T_m_i ;
        values[offset+2] = T_m_j+T_m_i+T_p_j ;
        values[offset+3] = -T_p_j ;

        {
          values[offset+2] += T_p_i;
        }
        offset += row_size ;
        ++irow ;
     }
  }
  {
    j=ny-1 ;
    {
      i=0 ;
      double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
      double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
      double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
      double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

      int row_size = 3 ;
      kcol[irow] = offset ;
      cols[offset] = irow-nx ;
      cols[offset+1] = irow ;
      cols[offset+2] = irow+1 ;
      values[offset] = -T_m_j ;
      values[offset+1] = T_m_j+T_p_i ;
      values[offset+2] = -T_p_i ;
      {
        values[offset+1] += T_m_i;
      }
      {
        values[offset+1] += T_p_j;
      }
      offset += row_size ;
      ++irow ;
    }
    for(i=1;i<nx-1;++i)
    {
      double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
      double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
      double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
      double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

      int row_size = 4 ;
      kcol[irow] = offset ;
      cols[offset] = irow-nx ;
      cols[offset+1] = irow-1 ;
      cols[offset+2] = irow ;
      cols[offset+3] = irow+1 ;
      values[offset] = -T_m_j ;
      values[offset+1] = -T_m_i ;
      values[offset+2] = T_m_j+T_m_i+T_p_i ;
      values[offset+3] = -T_p_i ;

      {
        values[offset+2] += T_p_j;
      }
      offset += row_size ;
      ++irow ;
    }
    {
        i=nx-1 ;
        double T_m_i = _trans_m_i(m_permitivity,i,j,nx,ny) ;
        double T_p_i = _trans_p_i(m_permitivity,i,j,nx,ny) ;
        double T_m_j = _trans_m_j(m_permitivity,i,j,nx,ny) ;
        double T_p_j = _trans_p_j(m_permitivity,i,j,nx,ny) ;

        int row_size = 3 ;
        kcol[irow] = offset ;
        cols[offset] = irow-nx ;
        cols[offset+1] = irow -1 ;
        cols[offset+2] = irow ;
        values[offset] = -T_m_j ;
        values[offset+1] = -T_m_i ;
        values[offset+2] = T_m_j+T_m_i ;
        {
          values[offset+2] += T_p_i;
        }
        {
          values[offset+2] += T_p_j;
        }
        offset += row_size ;
        ++irow ;
     }
  }
  kcol[irow] = offset ;
  //fprintf(stdout, "NROW : %d NNZ : %d\n",irow, offset) ;
}
