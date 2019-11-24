#include "objs/floyd_warshall.h"
#include "tbb/task_scheduler_init.h"
#include "tbb/tbb.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>

const int32_t INF = 1073741823;

void Write_file(const char *filename, int *AdjMatrix, const int &num_vertex) {
    std::ofstream out_file(filename);
    for (int32_t i = 0; i < num_vertex; i++) {
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
                    // ispc::floyd_warshall_simd(distance, n, k, i);
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
                // ispc::floyd_warshall_simd(distance, n, k, i);
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
    int32_t num_vertex(0), num_edge(0);
    int32_t src(0), dest(0), weight(0);

    file.read((char *)&num_vertex, sizeof(num_vertex));
    file.read((char *)&num_edge, sizeof(num_edge));
    int *AdjMatrix = new int[num_vertex * num_vertex]();

    std::fill_n(AdjMatrix, num_vertex * num_vertex, INF);
    for (int32_t i = 0; i < num_vertex; i++) {
        AdjMatrix[i * num_vertex + i] = 0;
    }

    int *tmp(new int[num_edge * 3]);
    file.read((char *)tmp, sizeof(int) * num_edge * 3);
    for (int32_t i = 0; i < num_edge; ++i) {
        src = tmp[i * 3];
        dest = tmp[i * 3 + 1];
        weight = tmp[i * 3 + 2];
        int idx(src * num_vertex + dest);
        AdjMatrix[idx] = weight;
    }
    file.close();
    floyd_warshall_omp(AdjMatrix, num_vertex);
    Write_file(argv[2], AdjMatrix, num_vertex);

    delete[](AdjMatrix);
    delete[](tmp);
    return 0;
}