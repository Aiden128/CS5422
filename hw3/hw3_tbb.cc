#include "objs/floyd_warshall.h"
#include "tbb/tbb.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <omp.h>
#include <cassert>

const int32_t MaxDistance = INT16_MAX;
const int32_t Out_max = 1073741823;
const int grain = 5;
int thread_count;

void Write_file(const char *filename, int *AdjMatrix, const int &num_vertex) {
    std::ofstream out_file(filename);
    for (int32_t i = 0; i < num_vertex; i++) {
        for (int32_t j = 0; j < num_vertex; j++) {
            if (AdjMatrix[i * num_vertex + j] == INT16_MAX) {
                // Write 2^30 -1
                out_file.write((char *)&Out_max, sizeof(Out_max));
            } else {
                // Write normal
                out_file.write((char *)&AdjMatrix[i * num_vertex + j], sizeof(int32_t));
            }
        }
    }
    out_file.close();
}

void floyd_warshall_parasimd(int *distance, size_t n) {
    for (size_t k=0; k < n; ++k) {
        tbb::parallel_for(
            tbb::blocked_range<int> {0, static_cast<int>(n), 20},
            [&] (auto r) {
                for (int i = r.begin(); i < r.end(); ++i) {
                    ispc::floyd_warshall_simd(distance, n, k, i);
                }
            }
        );
    }
}

void floyd_warshall_omp(int *distance, size_t n) {
    int32_t i, j, k;

    for (k = 0; k < n; k++) {
#pragma omp parallel num_threads(thread_count)
{
        int tid = omp_get_thread_num();
        int start = grain * tid;
        int end = tid == thread_count - 1 ? n : grain * (tid + 1);
        for (i = start; i < end; i++) {
            // for (j = 0; j < n; j++) {
            //     distance[i * n + j] = std::min(distance[i * n + j], (distance[i * n +k] + distance[k * n + j]));
            // }
            ispc::floyd_warshall_simd(distance, n, k, i);
        }
}
    }
}

void PrintData(int *AdjMatrix, const int &num_vertex) {
    for (int i = 0; i < num_vertex; ++i) {
        for (int j = 0; j < num_vertex; ++j) {
            std::cout << std::setw(6) << AdjMatrix[i * num_vertex + j];
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {

    thread_count = omp_get_num_threads();

    std::ifstream file(argv[1]);
    int32_t num_vertex(0), num_edge(0);
    int32_t src(0), dest(0), weight(0);

    file.seekg(0, std::ios_base::beg);
    file.read((char *)&num_vertex, sizeof(num_vertex));
    int *AdjMatrix = new int[num_vertex * num_vertex]();

    if (num_vertex < grain) thread_count = 1;

    for (int32_t i = 0; i < num_vertex; i++) {
        for (int32_t j = 0; j < num_vertex; j++) {
            int idx = i * num_vertex + j;
            if (__builtin_expect(i == j, false)) {
                AdjMatrix[idx] = 0;
            } else {
                AdjMatrix[idx] = MaxDistance;
            }
        }
    }

    file.read((char *)&num_edge, sizeof(num_edge));
    for (int32_t i = 0; i < num_edge; ++i) {
        file.read((char *)&src, sizeof(src));
        file.read((char *)&dest, sizeof(dest));
        file.read((char *)&weight, sizeof(weight));
        int idx(src * num_vertex + dest);
        AdjMatrix[idx] = weight;
    }
    file.close();
    floyd_warshall_omp(AdjMatrix, num_vertex);
    // if(__builtin_expect(num_vertex > 100, false)) {
    //     floyd_warshall_parasimd(AdjMatrix, num_vertex);
    // } else {
    //     ispc::floyd_warshall_seq(AdjMatrix, num_vertex);
    // }
    Write_file(argv[2], AdjMatrix, num_vertex);
    return 0;
}