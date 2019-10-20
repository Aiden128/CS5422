# CS5422 Homework 1 Report

107061517 張簡崇堯

[TOC]

## Implementation

### Data I/O

In my implementation, each rank has three local buffers, ``odd_buffer``, ``even_buffer`` and ``neighbor_buffer`` and a reference ``main_buffer`` to indicate the mainly used buffer of the rank.

In constructor, the ``main_buffer`` will point to ``odd_buffer`` or ``even_buffer`` according to the rank ID is odd or even.

I use MPI parallel I/O calls to read and write files simultaneously, the numbers read from the file will be store at ``main_buffer``.

### Sorting

After numbers are read, we need to perform one time sorting on ``main_buffer``, such that the merge function can works properly. I use the sorting algorithm from STL library in the beginning, I also tried Intel Parallel STL library in C++ 17, but the results are not satisfied (I will show the profiling results later).

Just one day before deadline, one of the classmates told me that using the ``float_sort`` from Boost library can significantly reduce the sorting time. Luckily, after applying Boost library, the total runtime is significantly reduced.

### Data exchange

In sequential version of odd-even sort, the abstraction level of **odd** or **even** is single number. However, if we use the same idea in parallel implementation, this approach will led to too many transmission calls with small data payload in flight. 

In my implementation, to enhance data throughput and reduce the number of transmission calls, the abstrction level is **MPI rank**. In each timestamp, the rank with its rank ID is even will communicate to its neighbor rank (rank ID + 1). Simultaneously, the rank with its rank ID is odd will communicate to its neighbor rank (rank. ID - 1).

The numbers from the neighbor ranks will be stored at ``neighbor_buffer``.

<img src="/Users/jerry/Documents/CS5422/hw1/README.assets/Untitled.png" alt="ra" style="zoom:50%;" />



###Merge function

After data exchange, the merge function will merge the numbers in ``main_buffer`` and ``neighbor_buffer``, and place the smaller numbers in ``odd_buffer`` (rank ID is even) or ``even_buffer`` (rank ID is odd). 

The pseudo code of the merge function is shown below (merge smaller numbers in this case)

```c
void merge (source1, source2, dest) {
    inputs source1, source2 : list
    output dest : list
    
    while(source1, source2 not empty && dest not full) {
        if (head(source1) < head(source2)) {
        	append head(source1) to dest
            drop head(source1)
        }
    }
    
    if(dest is not full) {
        if (source1 not empty) {
            fill dest with source1
        }
        else {
            fill dest with source2
        }
    }
}
```

### Status Synchronization

In order to know whether the sorting can be terminated or not, I use the ``MPI_Allreduce`` call to sync the sorting status.

## Experiment & Analysis

### 1. Methodology

#### System Spec

The testing environment is apollo cluster, provided by TA. 	

#### Performance Metrics

##### Odd-Even Sort

I use C++ ``chrono`` library to mesure the following items, and dump the measured items in YAML format and CSV format. 

To enable measurement, add ``-DPERF`` in compiler flags.

| Item             | Description                  |
| ---------------- | ---------------------------- |
| Mem              | Memory Allocation time       |
| Read             | MPI read file time           |
| Write            | MPI write file time          |
| MPI transmission | MPI send/receive calls' time |
| MPI sync         | MPI reduce calls' time       |
| Merge            | Merge function time          |
| Sort             | Sorting time                 |

The sum of the measured times is almost equal to the overall runtime (less than 10% error)

I use the seven indices above to generate the percentage stacked histogram, and apply *Amdahl's law* to analyze how can I optimize my codes.

##### STL Sort

I perform a profiling with C++ 17 Parallel STL and Boost library ``float_sort``, comparing the runtime of sorting a series of random number, the testing source code is accessible on my GitHub repository.

The range of input size is from 1 to 10,000,000, and each method is tested 100 times and take average as the final runtime.

## Experiment & Analysis

*   Sorting library comparison

    In this experiment, I try to examine which sorting function runs fastest.

![Sorting Comparison (/Users/jerry/Documents/CS5422/hw1/README.assets/Sorting Comparison (log-log scale)-1571211.png)](/Users/jerry/Downloads/Sorting Comparison (log-log scale).png)

The plot above shows that the ``float_sort`` function in Boost library has the smallest runtime in average compare to STL. Therefore, I use it in my implementation to achieve best performance.

## Experiences & Conclusion

A good library helps you a lot!

BTW, I think writing Makefile is too complicated, I think CMake can make things easier.