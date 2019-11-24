#include "objs/floyd_warshall.h"
#include "tbb/tbb.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>
#include <cassert>

const int32_t MaxDistance = INT16_MAX;
const int32_t Out_max = 1073741823;

class APSP {
  private:
    int32_t num_vertex;
    int *AdjMatrix;

  public:
    APSP() : num_vertex(0){};
    APSP(int32_t n);
    inline void AddEdge(int32_t from, int32_t to, int32_t weight);
    void PrintData();
    void Write_file(const char *filename);
    void FloydWarshall_omp();
    void FloydWarshall_tbb();
};

APSP::APSP(int32_t n) : num_vertex(n) {
    // Constructor, initialize AdjMatrix with 0 or MaxDistance
    AdjMatrix = new int[n * n]();
    for (int32_t i = 0; i < num_vertex; i++) {
        for (int32_t j = 0; j < num_vertex; j++) {
            int idx = i * n + j;
            if (i == j) {
                AdjMatrix[idx] = 0;
            } else {
                AdjMatrix[idx] = MaxDistance;
            }
        }
    }
}

void APSP::FloydWarshall_omp() {
int32_t i, j, k;

#pragma omp parallel shared(num_vertex, AdjMatrix) private(i, j, k) default(none)
    for (k = 0; k < num_vertex; k++) {
#pragma omp for schedule(dynamic)
        for (i = 0; i < num_vertex; i++) {
            for (j = 0; j < num_vertex; j++) {
                int idx = i * num_vertex + j;
                AdjMatrix[idx] = std::min(AdjMatrix[idx], (AdjMatrix[i * num_vertex + k] + AdjMatrix[k * num_vertex + j]));
            }
        }
    }
}

void APSP::FloydWarshall_tbb(){
    for (size_t k=0; k < num_vertex; ++k) {
        tbb::parallel_for(size_t(0), nun_vertex, size_t(1), [=](size_t i) {
            ispc::floyd_warshall_simd(&AdjMatrix, n, k, i); });
    }
}


inline void APSP::AddEdge(int32_t from, int32_t to, int32_t weight) {
    size_t idx(from * num_vertex + to);
    AdjMatrix[idx] = weight;
}

void APSP::PrintData() {
    for (int i = 0; i < num_vertex; ++i) {
        for (int j = 0; j < num_vertex; ++j) {
            std::cout << std::setw(6) << AdjMatrix[i * num_vertex + j];
        }
        std::cout << std::endl;
    }
}

void APSP::Write_file(const char *filename) {
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

int main(int argc, char **argv) {

    std::ifstream file(argv[1]);
    int32_t num_vertex(0), num_edge(0);
    int32_t src(0), dest(0), weight(0);

    file.seekg(0, std::ios_base::beg);
    file.read((char *)&num_vertex, sizeof(num_vertex));
    APSP apsp(num_vertex);
    file.read((char *)&num_edge, sizeof(num_edge));
    std::cout << "Vertex: " << num_vertex << " Edge: " << num_edge << std::endl;
    for (int32_t i = 0; i < num_edge; ++i) {
        file.read((char *)&src, sizeof(src));
        file.read((char *)&dest, sizeof(dest));
        file.read((char *)&weight, sizeof(weight));
        apsp.AddEdge(src, dest, weight);
    }
    file.close();
    apsp.FloydWarshall_omp();
    std::cout << "=======================" << std::endl;
    apsp.PrintData();
    apsp.Write_file(argv[2]);
    return 0;
}