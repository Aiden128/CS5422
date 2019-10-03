#ifndef OE_H
#define OE_H

#include <algorithm>
#include <string>
#include <iostream>
#include <exception>

class OE_sort {
public:
    OE_sort(int rank, int task_num, int total_size, 
            const char* input_file, const char* output_file);
    ~OE_sort();
private:
    int rank;
    int task_num;
    int total_size;
    int num_per_task;
    int res;
    int size;
    int offset;
    float* buffer;
    const char* input_file;
    const char* output_file; 
};

#endif