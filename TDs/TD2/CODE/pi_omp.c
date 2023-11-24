#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#include "omp.h"

#define TRIALS_PER_THREAD 10E10

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

    // TODO: initialisation du tableau stockant le résultat de chaque thread
    uint64_t* result = (uint64_t*)malloc(sizeof(uint64_t) * nb_threads);

    // TODO: initialisation du generateur de nombre pseudo aléatoires

    t0 = get_elapsedtime();

    // TODO: tirageS de flechettes

    // TODO: reduction des resultats
    t1 = get_elapsedtime();
    duration = (t1 - t0);
    fprintf(stdout, "%ld of %ld throws are in the circle ! (Time: %lf s)\n",
            (uint64_t)pi, n_test, duration);
    // TODO: estimation de Pi
    fprintf(stdout, "Pi ~= %lf\n", pi);

    return 0;
}
