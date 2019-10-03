#include <iostream>
#include <mpi.h>
#include <cassert>
#include <vector>
#include <numeric>
#include <iomanip>

using namespace std;

template <typename T > 
void print_vec(vector<T> vec){
    for (auto i : vec){
        cout << i << " ";
    }
    cout << endl;
}

enum para_mode{
    single,
    divide
};


int main(int argc, char **argv){
    int rank(0), task_num(0), float_num(0);
    MPI_File in_file;
    MPI_Offset in_file_size;
    para_mode mode;

    // Start processing
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert(MPI_File_open(MPI_COMM_WORLD, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &in_file) == MPI_SUCCESS);
    assert(MPI_File_get_size(in_file, &in_file_size) == MPI_SUCCESS);

    vector<int> read_schedule(task_num);
    vector<int> read_start(task_num), read_end(task_num);
    

    // Parallel read schedule
    if(rank == 0) {
        assert(in_file_size % sizeof(float) == 0);
        float_num = in_file_size / sizeof(float);

        if (float_num > task_num) {
            mode = divide;
            int num_per_task = float_num / task_num;
            int res = float_num % task_num;

            read_schedule[0] = num_per_task + res;
            for(int i = 1; i < task_num; ++i) {
                read_schedule[i] = num_per_task;
            }
            for(int i = 0, read_count = 0; i < task_num; ++i){
            read_end[i] = read_count + read_schedule[i] - 1;
            read_count += read_schedule[i];
            }
            for (int i = 0; i < task_num; ++i){
                read_start[i] = read_end[i] - read_schedule[i] + 1;
            }
        } else { // Don't divide, do on single thread
            mode = single;
            read_schedule[0] = float_num;
            read_start[0] = 0;
            read_end[0] = float_num - 1;
        }

        // Debug section: Print schedule
        cout << "Mode = " << mode << endl;
        cout << "Number of tasks = "<< task_num << endl;
        cout << "File size (bytes) = " << in_file_size << endl;
        cout << "Number of float = " << float_num <<endl;
        cout << "Schedule of each thread" << endl;
        print_vec(read_schedule);
        cout << "Read start "<< endl;
        print_vec(read_start);
        cout << "Read end "<< endl;
        print_vec(read_end);

    }
    



    MPI_File_close(&in_file);
    MPI_Finalize();
    return 0;

}