#ifndef OE_H
#define OE_H
#ifdef PERF
#include <chrono>
#include <fstream>
#include <numeric>
#include <string>
#endif

#include <algorithm>
#include <boost/sort/spreadsort/float_sort.hpp>
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

    bool _left();
    bool _right();
    void _merge_small(float *src1, size_t src1_size, float *src2,
                      size_t src2_size, float *dest, size_t dest_size);
    void _merge_large(float *src1, size_t src1_size, float *src2,
                      size_t src2_size, float *dest, size_t dest_size);
};

#endif