// CS5540 Lab1
// Author: Jerry ZJ
// Date: 2019/09/26
#include <cmath>
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv) {
    MPI_Status status;
    const long long N(static_cast<long long>(atoll(argv[1])));
    int rank(0), task_num(0);
    double area(0.0), x(0.0), temp(0.0);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);

    // Devide N slices to n different tasks
    const int N_per_task(N / task_num), Res(N % task_num);

    if (rank == 0) {
        // rank 0 will handle remain parts
        for (int i = 0; i < N_per_task + Res; ++i) {
            x = i;
            area += sqrt(1 - pow((x / N), 2.0));
        }
        // Collect results
        for (int i = 1; i < task_num; ++i) {
            MPI_Recv(&temp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            area += temp;
        }
        cout.precision(15);
        cout << static_cast<double>(area * 4.0 / N) << endl;
    } else {
        double area_temp(0.0);

        for (int i = 0; i < N_per_task; ++i) {
            x = rank * N_per_task + Res + i;
            area_temp += sqrt(1 - pow((x / N), 2.0));
        }
        MPI_Request req;
        MPI_Isend(&area_temp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
    }

    MPI_Finalize();
    return 0;
}