#include "oe.h"

OE_sort::OE_sort(int rank, int task_num, int total_size, 
            const char* input_file, const char* output_file)
            : rank(rank), task_num(task_num), total_size(total_size),
            input_file(input_file), output_file(output_file) {
                num_per_task = total_size / task_num;
                res = total_size / task_num; // remaining parts handled by rank0
                offset = num_per_task * rank + std::min(rank, res);
                buffer = (new float[size + 2]);
}

OE_sort::~OE_sort(){delete[](buffer);}