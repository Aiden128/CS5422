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

bool OE_sort::_do_left() {

}

bool OE_sort::_do_right() {
    
}

void OE_sort::_merge_small(Buffer &src1, Buffer &src2, Buffer &dest) {
    // Set iter to index 0
    src1.reset_iter(src1.head);
    src2.reset_iter(src2.head);
    dest.reset_iter(dest.head);

    // Fill data
    while(src1.iter < src1.end && src2.iter < src2.end && dest.iter < dest.end){
        if (*src1.iter < *src2.iter) {
            *dest.iter++ = *src1.iter++;
        }
        else {
            *dest.iter++ = *src2.iter++;
        }
    }

    // Fill blanks in dest
    if (dest.iter < dest.end){
        int offset(0);
        if(src1.iter < src1.end){
            offset = std::min(dest.end - dest.iter, src1.end - src1.iter);
            std::copy(src1.iter, src1.iter + offset, dest.iter);
        }
        else{
            offset = std::min(dest.end - dest.iter, src2.end - src2.iter);
            std::copy(src2.iter, src2.iter + offset, dest.iter);
        }
    }
}

void OE_sort::_merge_large(Buffer &src1, Buffer &src2, Buffer &dest) {
    // Set iter to the end of buffer
    src1.reset_iter(src1.tail);
    src2.reset_iter(src2.tail);
    dest.reset_iter(dest.tail);

    // Fill data
    while(src1.base <= src1.iter && src2.base <= src2.iter && dest.base <= dest.iter){
        if (*src1.iter > *src2.iter) {
            *dest.iter-- = *src1.iter--;
        }
        else {
            *dest.iter-- = *src2.iter--;
        }
    }

    // Fill blanks in dest
    if (dest.base < dest.iter){
        int offset(0);
        if(src1.iter < src1.end){
            offset = std::min(dest.iter - dest.base, src1.iter - src1.base) + 1;
            std::copy(src1.iter, src1.iter + offset, dest.iter);
        }
        else{
            offset = std::min(dest.iter - dest.base, src2.iter - src2.base) + 1;
            std::copy(src2.iter, src2.iter + offset, dest.iter);
        }
    }
}