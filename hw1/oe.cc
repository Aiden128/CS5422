#include "oe.h"

using namespace std;

OE_sort::OE_sort(int rank, int task_num, int file_size, const char *input_file,
                 const char *output_file)
    : rank(rank), task_num(task_num), file_size(file_size),
      input_file(input_file),
      output_file(output_file), global_sorted(false), local_sorted(false) {
    num_per_task = file_size / task_num;
    res = file_size % task_num; // remaining parts
    offset = num_per_task * rank + std::min(rank, res);
    size = num_per_task + (rank < res); // remaining parts will divided by all
    // Calculate left/right buffer size
    if(rank == 0){ left_size = 0;}
    else{
        left_size = num_per_task + ((rank - 1) < res);
    }
    if((rank + 1) == task_num) { right_size = 0;}
    else {
        right_size = num_per_task + ((rank + 1) < res);
    }
    buffer = (new float[size]);
    // Debug
    //cout << "Rank = "<<rank<< " Size = "<<size<<" lSize = " << left_size << " rSize = "<< right_size <<endl;
}

OE_sort::~OE_sort() { delete[](buffer); }

void OE_sort::read_file() {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &fh);
    MPI_File_read_at_all(fh, offset * sizeof(float), buffer, size, MPI_FLOAT,
                         MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
    // debug
    //cout << "rank = "<<rank << endl;
    //cout << "Content" <<endl;
    //for(int i = 0; i < size; ++i){
    //    cout << buffer[i] << " ";
    //}
    //cout << endl;
}

void OE_sort::write_file() {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, output_file,
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    MPI_File_write_at_all(fh, offset * sizeof(float), buffer, size, MPI_FLOAT,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

bool OE_sort::odd_even(int state) {
    bool sorted(true);
    MPI_Request l_req, r_req;

    int i(0), end(0);
    for(; i < end; i+=2){
        if(buffer[i] > buffer[i+1]){
            sorted = false;
            std::swap(buffer[i], buffer[i+1]);
        }
    }
    return sorted;
}

void OE_sort::sort() {
    while (!global_sorted) {
        global_sorted = true;
        local_sorted = local_sorted & odd_even(0);
        local_sorted = local_sorted & odd_even(1);
        // Question: MPI datatype for bool???
        MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                      MPI_COMM_WORLD);
    }
}