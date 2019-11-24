#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include <cstdint>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>

const int INF = 1073741823;

void Write_file(const char *filename, int *AdjMatrix, const int &num_vertex) {
    std::ofstream out_file(filename);
    for (int i = 0; i < num_vertex; i++) {
        out_file.write((char *)&AdjMatrix[i * num_vertex],
                       sizeof(int) * num_vertex);
    }
    out_file.close();
}

void floyd_warshall_parasimd(int *distance, size_t n) {
    int thread_count = omp_get_max_threads();
    tbb::task_scheduler_init init(thread_count);
    int grain_size = n / thread_count;
    for (size_t k = 0; k < n; ++k) {
        tbb::parallel_for(
            tbb::blocked_range<int>{0, static_cast<int>(n),
                                    static_cast<size_t>(grain_size)},
            [&](auto r) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    for (int j = 0; j < n; ++j) {
                        distance[i * n + j] = std::min(
                            distance[i * n + j],
                            (distance[i * n + k] + distance[k * n + j]));
                    }
                }
            });
    }
}

void floyd_warshall_omp(int *distance, size_t n) {
    int thread_count = omp_get_max_threads();
    int grain(n / thread_count);
    for (int k = 0; k < n; k++) {
#pragma omp parallel num_threads(thread_count)
        {
            int tid = omp_get_thread_num();
            int start = grain * tid;
            int end = tid == thread_count - 1 ? n : grain * (tid + 1);
            for (int i = start; i < end; ++i) {
                for (int j = 0; j < n; ++j) {
                    distance[i * n + j] =
                        std::min(distance[i * n + j],
                                 (distance[i * n + k] + distance[k * n + j]));
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    std::ifstream file(argv[1]);
    int num_vertex(0), num_edge(0);
    int src(0), dest(0), weight(0);

    auto read_start = std::chrono::high_resolution_clock::now();

    file.read((char *)&num_vertex, sizeof(num_vertex));
    file.read((char *)&num_edge, sizeof(num_edge));
    int *AdjMatrix = new int[num_vertex * num_vertex]();

    std::fill_n(AdjMatrix, num_vertex * num_vertex, INF);
    for (int i = 0; i < num_vertex; i++) {
        AdjMatrix[i * num_vertex + i] = 0;
    }

    int *tmp(new int[num_edge * 3]);
    file.read((char *)tmp, sizeof(int) * num_edge * 3);
    for (int i = 0; i < num_edge; ++i) {
        src = tmp[i * 3];
        dest = tmp[i * 3 + 1];
        weight = tmp[i * 3 + 2];
        int idx(src * num_vertex + dest);
        AdjMatrix[idx] = weight;
    }
    file.close();

    auto read_end = std::chrono::high_resolution_clock::now();
    auto read_time = std::chrono::duration_cast<std::chrono::nanoseconds>(read_end - read_start).count();

    auto fw_start = std::chrono::high_resolution_clock::now();
    floyd_warshall_omp(AdjMatrix, num_vertex);
    auto fw_end = std::chrono::high_resolution_clock::now();
    auto fw_time = std::chrono::duration_cast<std::chrono::nanoseconds>(fw_end - fw_start).count();


    auto write_start = std::chrono::high_resolution_clock::now();
    Write_file(argv[2], AdjMatrix, num_vertex);
    auto write_end = std::chrono::high_resolution_clock::now();
    auto write_time = std::chrono::duration_cast<std::chrono::nanoseconds>(write_end - write_start).count();

    std::ofstream perf_file;
    perf_file.open("profile.csv", std::ofstream::out | std::ofstream::app);
    perf_file << read_time << ",";
    perf_file << fw_time << ",";
    perf_file << write_time << "," << std::endl;
    perf_file.close();

    delete[](AdjMatrix);
    delete[](tmp);
    return 0;
}