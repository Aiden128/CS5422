#ifndef OE_H
#define OE_H

#include <algorithm>
#include <exception>
#include <iostream>
#include <mpi.h>

enum schedule_mode { single, parallel, parallel_thr, null };

class OE_sort {
  public:
    explicit OE_sort(int rank, int task_num, int file_size,
                     const char *input_file, const char *output_file);
    ~OE_sort();
    void parallel_read_file();
    void parallel_write_file();
    void parallel_sort();
    void single_read_file();
    void single_write_file();
    void single_sort();
    schedule_mode schedule;

  private:
    enum mpi_tags { left, right, null };
    const int rank;     // My rank ID
    const int task_num; // Total ranks
    float *&main_buffer;
    const char *input_file;
    const char *output_file;

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