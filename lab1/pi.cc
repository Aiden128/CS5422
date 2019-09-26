#include <iostream>
#include <mpi.h>
#include <cmath>
#include <cassert>

using namespace std;

int main(int argc, char** argv){

    assert(argc == 2);

    MPI_Status status;
    const long long N(static_cast<long long>(atoll(argv[1])));
    const double N_inv(static_cast<double>(1.0 / N));
    int rank(0);
    int task_num(0);
    int rc(0);
    double pi(0.0);
    double area(0.0);
    double x(0.0);
    double y(0.0);
    double temp(0.0);

    rc = MPI_Init(&argc, &argv);
    if(rc != MPI_SUCCESS){
        cerr << "MPI_Init returned nonzero" << endl;
        exit(-1);
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);

    int N_per_task = N / task_num;
    int Res = N % task_num;

    if(rank == 0){
        for (int i = 0; i < N_per_task + Res; ++i){
            x = i;
            y = sqrt(1 - pow((x / N), 2.0));
            area += static_cast<double>(y);
        }
        for (int i = 1; i < task_num; ++i){
            MPI_Recv(&temp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            area += temp;
        }
        pi = area * 4 * N_inv;
        cout.precision(15);
        cout<<pi<<endl;
    }
    else{
        double area_temp(0.0);

        for (int i = 0; i < N_per_task; ++i){
            x = rank * N_per_task + i + Res;
            y = sqrt(1 - pow((x / N), 2.0));
            area_temp += static_cast<double>(y);
        }
        MPI_Request req;
        MPI_Isend(&area_temp, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req);
    }

    MPI_Finalize();
    return 0;
}
