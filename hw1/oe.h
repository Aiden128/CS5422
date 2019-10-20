#ifndef OE_H
#define OE_H

#ifdef PERF
#include <chrono>
#include <fstream>
#include <numeric>
#include <string>
#endif

#include "tbb/task_scheduler_init.h"
#include <boost/sort/spreadsort/float_sort.hpp>
#include <algorithm>
#include <exception>
#include <execution>
#include <iostream>
#include <mpi.h>

enum schedule_mode { single, fuse, parallel, null };

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

#ifdef PERF
    double mem_time;
    double read_time;
    double write_time;
    double MPI_transmission_time;
    double MPI_sync_time;
    double merge_time;
    double stl_sort_time;

    double avg_mem;
    double avg_read;
    double avg_write;
    double avg_trans;
    double avg_sync;
    double avg_merge;
    double avg_stl_sort;
#endif

private:
    enum mpi_tags { left, right, null };
    const int rank;     // My rank ID
    const int task_num; // Total ranks
    float *&main_buffer;
    const char *input_file;
    const char *output_file;
    tbb::task_scheduler_init task_scheduler;
    MPI_Request request;

    int num_per_task; // Number of floats handled by a rank
    int res;          // Remaining part
    int size;         // Buffer size, num of float
    int offset;       // Read starting point
    int left_size;    // Data number of left rank
    int right_size;   // Data number of right rank
    float *neighbor_buffer;
    float *even_buffer;
    float *odd_buffer;

    bool _do_left();
    bool _do_right();
    void _merge_small(float *src1, size_t src1_size, float *src2,
                      size_t src2_size, float *dest, size_t dest_size);
    void _merge_large(float *src1, size_t src1_size, float *src2,
                      size_t src2_size, float *dest, size_t dest_size);
};

#endif