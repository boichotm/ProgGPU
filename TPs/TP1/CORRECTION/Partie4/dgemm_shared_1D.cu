#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

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

    __global__
void MulMatrixKernel(double* A, double* B, double* C, int N)
{
    int col		= threadIdx.x + blockDim.x * blockIdx.x;
    int ligne	= threadIdx.y + blockDim.y * blockIdx.y;

    if((col < N) && (ligne < N))
    {
        double val = 0.0f;
        for(int k = 0; k < N; k++)
        {
            val += A[ligne * N + k] * B[k * N + col];
        }
        C[ligne * N + col] = val;
    }
}

__global__
void MulMatrixShare(double* A, double* B, double* C, int N){

    // tuile en shared memory sous format row major
    // dimensions de la tuile = dimension du block de thread
    __shared__ double A_s[BLOCK_WIDTH * BLOCK_WIDTH];
    __shared__ double B_s[BLOCK_WIDTH * BLOCK_WIDTH];

    // indices des premières ligne et colonne calculées par le bloc de thread
    int blockRow = BLOCK_WIDTH * blockIdx.y;
    int blockCol = BLOCK_WIDTH * blockIdx.x;

    // indices locales à la tuile
    int threadCol = threadIdx.x % BLOCK_WIDTH;
    int threadRow = threadIdx.x % BLOCK_WIDTH;

    // indices de la valeur de C calculé par le thread

    double value = 0.0f;

    for(int tile_offset = 0; tile_offset < N; tile_offset+=BLOCK_WIDTH)
    {

        A_s[BLOCK_WIDTH * threadRow + threadCol] = A[(blockRow /* première ligne traitée par le bloc de thread */
                + threadRow) * N /* décalage à la ligne traitée par le thread */ 
            + tile_offset  /* décalage  à la tuile courante */
            + threadCol]; /* dcalage à la colonne traitée par le thread */

        B_s[BLOCK_WIDTH * threadRow + threadCol] = B[tile_offset * N /* décalage à la ligne tuile courante */
            + threadRow * N /* décalage à la ligne du thread */
            + blockCol /* décalage à la colonne du bloc */
            + threadCol]; /* décalage à la colonne du thread */

        // Attente que tous les threads ont bien chargé dans la mémoire partagée leurs deux indices
        __syncthreads();

        for(int k =0; k < BLOCK_WIDTH; k++)
        {
            value += A_s[threadRow * BLOCK_WIDTH + k] * B_s[k * BLOCK_WIDTH + threadCol];
        }

        // S'assurer que tous les threads ont bien fini le calcul du préliminaire du tile courant avant de commencer la prochaine étape du calcul de cette tile
        __syncthreads();
    }

    // Enregistrer la valeur accumulée dans C (mémoire globale)
    C[(blockRow + threadRow) * N  + blockCol + threadCol] = value;
}

int main(int argc, char** argv)
{
    int N, use_cpu;

    N = (argc < 2)?1000:atoi(argv[1]);
    use_cpu = (argc < 3)?0:atoi(argv[2]);

    double t0 = 0., t1 = 0., duration = 0.;

    int nbBlocks = N / BLOCK_WIDTH;
    //if(N % BLOCK_WIDTH) nbBlocks++;
    if(N % BLOCK_WIDTH) N += (N % BLOCK_WIDTH);
    dim3 gridSize(nbBlocks, nbBlocks);
    dim3 blockSize(BLOCK_WIDTH * BLOCK_WIDTH);

    double *A, *B, *C, *C_ref;
    double *d_A, *d_B, *d_C;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    A = (double*) malloc(sizeof(double) * N * N);
    B = (double*) malloc(sizeof(double) * N * N);
    C = (double*) malloc(sizeof(double) * N * N);
    if (use_cpu)
        C_ref = (double*) malloc(sizeof(double) * N * N);

    cudaMalloc(&d_A, sizeof(double) * N * N);
    cudaMalloc(&d_B, sizeof(double) * N * N);
    cudaMalloc(&d_C, sizeof(double) * N * N);

    // Value initialization
    init(A, B, C, N);

    cudaMemcpy(d_A, A, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(double) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(double) * N * N, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    //MulMatrixKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    MulMatrixShare<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    cudaMemcpy(C, d_C, sizeof(double) * N * N, cudaMemcpyDeviceToHost);

    // Compute multiplication on cpu
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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A);
    free(B);
    free(C);

    return 0;
}
