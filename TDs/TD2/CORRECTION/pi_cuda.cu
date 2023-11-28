#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <curand_kernel.h>

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

/* QUESTION 3 */
#define TRIALS_PER_THREAD 4096
#define BLOCKS 512
#define THREADS 256
/* FIN QUESTION 3*/

/* QUESTION 6 */
__global__ void gpu_monte_carlo(float *estimate, curandState *states) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int points_in_circle = 0;
  float x = 0., y = 0.;

  curand_init(2020, tid, 0, &states[tid]);  //   Initialize CURAND

  for(int i = 0; i < TRIALS_PER_THREAD; i++)
  {
    x = curand_uniform (&states[tid]);
    y = curand_uniform (&states[tid]);
    points_in_circle += (x*x + y*y <= 1.0f); // count if x & y is in the circle.
  }
  estimate[tid] = 4.0f * points_in_circle / (float) TRIALS_PER_THREAD; // return estimate of pi
}
/* FIN QUESTION 6 */

int main (int argc, char *argv[]) {
  float h_counts[BLOCKS * THREADS] = { 0 };
  double t0 = 0., t1 = 0., duration = 0.;

  printf("# of trials per thread = %d, # of blocks = %d, # of threads/block = %d.\n", TRIALS_PER_THREAD, BLOCKS, THREADS);

/* QUESTION 4 */
  float *d_counts;
  cudaMalloc((void **) &d_counts, BLOCKS * THREADS * sizeof(float)); // allocate device mem. for counts
/* FIN QUESTION 4 */

/* QUESTION 6 */
  curandState *d_states;
  cudaMalloc( (void **)&d_states, THREADS * BLOCKS * sizeof(curandState) );
/* FIN QUESTION 6 */

  t0 = get_elapsedtime();
/* QUESTION 3 */
  gpu_monte_carlo<<<BLOCKS, THREADS>>>(d_counts, d_states);
/* FIN QUESTION 3*/

/* QUESTION 5 */
  cudaMemcpy(h_counts, d_counts, BLOCKS * THREADS * sizeof(float), cudaMemcpyDeviceToHost); // return results 
/* FIN QUESTION 5 */

  float pi_gpu = 0.f;
  for(int i = 0; i < BLOCKS * THREADS; i++)
  {
    pi_gpu += h_counts[i];
  }

  pi_gpu /= (BLOCKS * THREADS);

  t1 = get_elapsedtime();
  duration = (t1 - t0);

  printf("GPU pi calculated in %lf s.\n", duration);
  fprintf(stdout, "Pi ~= %lf\n", pi_gpu);

  return 0;
}
