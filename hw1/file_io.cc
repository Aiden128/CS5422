#include <iostream>
#include <mpi.h>
#include <cassert>
#include <vector>
#include <numeric>

using namespace std;

int main(int argc, char **argv){
    int rank(0), task_num(0), float_num(0);
    MPI_File in_file;
    MPI_Offset in_file_size;
    

    // Start processing
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert(MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file) == MPI_SUCCESS);
    assert(MPI_File_get_size(in_file, &in_file_size) == MPI_SUCCESS);

    vector<int> read_schedule(task_num);
    vector<int> read_start(task_num), read_end(task_num);
    cout << "Number of tasks = "<< task_num << endl;
    cout << "File size (bytes) = " << in_file_size << endl;
    assert(in_file_size % sizeof(float) == 0);
    float_num = in_file_size / sizeof(float);
    cout << "Number of float = " << float_num <<endl;

    // Parallel read schedule
    if(rank == 0) {
        if (float_num > task_num) {
            int num_per_task = float_num / task_num;
            int res = float_num % task_num;

            read_schedule[0] = num_per_task + res;
            for(i = 1; i < task_num; ++i) {
                read_schedule[i] = num_per_task;
            }
        } else {
            read_schedule[0] = float_num;
        }
        std::partial_sum(schedule.begin(), schedule.end(), read_end.begin(), plus<int>());
        for (int i = 0; i < task_num; ++i){
            read_start[i] = read_end[i] - schedule[i];
        }
    }




    MPI_File_close(&in_file);
    MPI_Finalize();
    return 0;

}