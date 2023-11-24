#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    uint64_t n_test = 10E7;
    uint64_t i;
    uint64_t count = 0;
    double x = 0., y = 0.;
    double pi = 0.;

    // TODO: inialisation du generateur de nombres pseudo aleatoires.
    // TODO: tirageS de flechettes

    fprintf(stdout, "%ld of %ld throws are in the circle !\n", count, n_test);
    // TODO: estimation de Pi
    fprintf(stdout, "Pi ~= %lf\n", pi);

    return 0;
}
