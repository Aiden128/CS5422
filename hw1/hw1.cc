// CS5540 Homework 1
// Author: Jerry ZJ
// Date: 2019/09/27
#include <iostream>
#include <mpi.h>

using namespace std;

int main(int argc, char **argv) {
    int rank(0), task_num(0);
    MPI_File f;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    
    MPI_File_open(MPI_COMM_WORLD, argv[1], MPI_MODE_RDONLY, MPI_INFO_NULL, &f);
    float data[1];
    MPI_File_read_at(f, sizeof(float) * rank, &data, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    printf("rank %d got float: %f\n", rank, data);
    //MPI_File_write_at(f, ...

    MPI_Finalize();
    return 0;
}