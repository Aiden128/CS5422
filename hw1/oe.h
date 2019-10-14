#ifndef OE_H
#define OE_H

#include <algorithm>
#include <exception>
#include <iostream>
#include <mpi.h>

class OE_sort {
  public:
    explicit OE_sort(int rank, int task_num, int file_size,
                     const char *input_file, const char *output_file);
    ~OE_sort();
    void read_file();
    void write_file();
    void sort();

  private:
    enum mpi_tags { left, right, null };
    int rank;      // My rank ID
    int task_num;  // Total ranks
    int file_size; // File size, should be N * sizeof(float)
    float *&curr_buffer;
    const char *input_file;
    const char *output_file;
    bool global_sorted;
    bool local_sorted;

    int num_per_task; // Number of floats handled by a rank
    int res;          // Remaining part
    int size;         // Buffer size, num of float
    int offset;       // Read starting point
    int left_size;    // Data number of left rank
    int right_size;   // Data number of right rank
    float *neighbor_buffer;
    float *buffer0;
    float *buffer1;

    bool _do_left();
    bool _do_right();
    void _merge_small(float *src1, ssize_t src1_size, float *src2,
                      ssize_t src2_size, float *dest, ssize_t dest_size);
    void _merge_large(float *src1, ssize_t src1_size, float *src2,
                      ssize_t src2_size, float *dest, ssize_t dest_size);
};

#endif