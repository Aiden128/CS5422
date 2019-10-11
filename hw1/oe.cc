#include "oe.h"

using namespace std;

OE_sort::OE_sort(int rank, int task_num, int file_size, const char *input_file,
                 const char *output_file)
    : rank(rank), task_num(task_num), file_size(file_size),
      input_file(input_file),
      output_file(output_file), global_sorted(false), local_sorted(false) {
    
    // Data partition
    num_per_task = file_size / task_num;
    res = file_size % task_num; 
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

    neighbor_buffer = new float[std::max(left_size, right_size)];
    buffer0 = new float[size];
    buffer1 = new float[size];
    if(rank % 2){
        curr_buffer = buffer0;
    }
    else {
        curr_buffer = buffer1;
    }
    // Debug
    //cout << "Rank = "<<rank<< " Size = "<<size<<" lSize = " << left_size << " rSize = "<< right_size <<endl;
}

OE_sort::~OE_sort() { 
    delete[](buffer0);
    delete[](buffer1);
    delete[](neighbor_buffer);
}

void OE_sort::read_file() {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &fh);
    MPI_File_read_at_all(fh, offset * sizeof(float), curr_buffer, size, MPI_FLOAT,
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
    MPI_File_write_at_all(fh, offset * sizeof(float), curr_buffer, size, MPI_FLOAT,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void OE_sort::sort() {
    // use STL to sort local content
    std::sort(curr_buffer, curr_buffer + size);
    local_sorted = true;
    // Split odd rank & even rank
    if (rank % 2) {
        while(!global_sorted){
            local_sorted = local_sorted & do_left();
            local_sorted = local_sorted & do_right();
            // Sync sorting status
            MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);
        }
    }
    else {
        while(!global_sorted){
            local_sorted = local_sorted & do_right();
            local_sorted = local_sorted & do_left();
            MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);
        }
    }
}

bool OE_sort::do_left() {

}

bool OE_sort::do_right() {
    
}