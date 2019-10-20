#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
int main (int argc, char **argv) {
    int rank, size, i;
    double pi=0.0, area=0.0, sum=0.0, x = 0.0;

    int N = atoi(argv[1]);
    MPI_Init(&argc, &argv);
    MPI_Comm_rank (MPI_COMM_WORLD, &rank);
    MPI_Comm_size (MPI_COMM_WORLD, &size);

#pragma omp parallel for reduction(+:area) private(x)
    for (i = rank; i < N; i += size) {
        x=(double)i*(double)i/((double)N*(double)N);
        area += sqrt(1-x)/N;
    }
    
    MPI_Reduce(&area, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        pi = 4.0 * sum;
        printf("%.10f\n",pi);
    }

    return 0;
}

