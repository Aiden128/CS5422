#include "oe.h"

using namespace std;

OE_sort::OE_sort(int rank, int task_num, int file_size, const char *input_file,
                 const char *output_file)
    : rank(rank), task_num(task_num), file_size(file_size),
      input_file(input_file), output_file(output_file) {
    num_per_task = file_size / task_num;
    res = file_size % task_num; // remaining parts handled by rank0
    offset = num_per_task * rank + std::min(rank, res);
    size = num_per_task + (rank < res);
    // Calculate left/right buffer size

    cout << "rank = " << rank << " size = " << size << endl;
    buffer = (new float[size + 2]);
}

OE_sort::~OE_sort() { delete[](buffer); }

void OE_sort::read_file() {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &fh);
    MPI_File_read_at_all(fh, offset * sizeof(float), buffer, size, MPI_FLOAT,
                         MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void OE_sort::write_file() {
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, output_file,
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    MPI_File_write_at_all(fh, offset * sizeof(float), buffer, size, MPI_FLOAT,
                          MPI_STATUS_IGNORE);
    MPI_File_close(&fh);
}

void OE_sort::sort() {
    
}