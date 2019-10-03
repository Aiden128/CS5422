// CS5540 Homework 1: Odd-Even Sort
// Author: Jerry ZJ
// Date: 2019/09/27
#include "util.h"
#include "oe.h"
#include <iostream>
#include <mpi.h>
#include <cassert>

using namespace std;

int main(int argc, char **argv) {

    assert(argc == 4); // Make sure arguments are ready

    int rank(0), task_num(0);
    // Start processing
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    OE_sort oe(rank, task_num, stoi(argv[1]), argv[2], argv[3]);
    //oe.sort();

    MPI_Finalize();
    return 0;
}