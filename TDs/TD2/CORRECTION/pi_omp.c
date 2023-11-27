#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "omp.h"

#define TRIALS_PER_THREAD 10E7

#define gettime(t) clock_gettime(CLOCK_MONOTONIC_RAW, t)
#define get_sub_seconde(t) (1e-9 * (double)t.tv_nsec)
/** return time in second
 */
double get_elapsedtime(void) {
    struct timespec st;
    int err = gettime(&st);
    if (err != 0) return 0;
    return (double)st.tv_sec + get_sub_seconde(st);
}

int main(int argc, char** argv) {
    uint64_t const n_test = TRIALS_PER_THREAD;
    uint64_t i;
    double x = 0., y = 0.;
    double pi = 0.;
    double t0 = 0., t1 = 0., duration = 0.;

    int nb_threads = 0;
#pragma omp parallel shared(nb_threads)
#pragma omp master
    nb_threads = omp_get_num_threads();
    fprintf(stdout, "Nb threads: %d\n", nb_threads);

    uint64_t* result = (uint64_t*)malloc(sizeof(uint64_t) * nb_threads);
    for (i = 0; i < nb_threads; ++i) {
        result[i] = 0;
    }

    srand(2023);
    t0 = get_elapsedtime();
#pragma omp parallel shared(result) firstprivate(n_test, x, y)
    {
        uint64_t local_res = 0;
        unsigned int seed = time(NULL) ^ omp_get_thread_num() ^ getpid();
#pragma omp for schedule(static)
        for (i = 0; i < n_test; ++i) {
            x = rand_r(&seed) / (double)RAND_MAX;
            y = rand_r(&seed) / (double)RAND_MAX;
            local_res += (((x * x) + (y * y)) <= 1);
        }
        int tid = omp_get_thread_num();
        result[tid] = local_res;
    }

    for (i = 0; i < nb_threads; ++i) {
        pi += result[i];
    }
    t1 = get_elapsedtime();
    duration = (t1 - t0);
    fprintf(stdout, "%ld of %ld throws are in the circle ! (Time: %lf s)\n",
            (uint64_t)pi, n_test, duration);
    pi *= 4;
    pi /= (double)n_test;
    fprintf(stdout, "Pi ~= %lf\n", pi);

    return 0;
}
