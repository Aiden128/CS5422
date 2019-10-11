#ifndef OE_H
#define OE_H

#include <algorithm>
#include <exception>
#include <iostream>
#include <mpi.h>
#include <string>

class OE_sort {
public:
    OE_sort(int rank, int task_num, int file_size, const char *input_file,
            const char *output_file);
    ~OE_sort();
    void read_file();
    void write_file();
    void sort();

private:
    int rank;         // My rank ID
    int task_num;     // Total ranks
    int file_size;    // File size, should be N * sizeof(float)
    int num_per_task; // Number of floats handled by a rank
    int res;          // Remaining part
    int size;         // Buffer size, num of float
    int left_size;    // Data number of left rank
    int right_size;   // Data number of right rank
    int offset;       // Read starting point
    float *&curr_buffer;
    float *buffer0;
    float *buffer1;
    float *neighbor_buffer;
    bool global_sorted;
    bool local_sorted;
    const char *input_file;
    const char *output_file;

    bool do_left();
    bool do_right();
};

#endif