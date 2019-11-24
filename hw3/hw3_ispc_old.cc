#include "objs/floyd_warshall.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>

const int32_t MaxDistance = INT16_MAX;
const int32_t Out_max = 1073741823;
int32_t block_size = 16;

class APSP {
  private:
    int32_t num_vertex;
    int32_t n_oversized;
    int *AdjMatrix;

  public:
    APSP() : num_vertex(0){};
    APSP(int32_t n);
    inline void AddEdge(int32_t from, int32_t to, int32_t weight);
    void PrintData();
    void Write_file(const char *filename);
    //void FloydWarshall();
    void FloydWarshall(const int &n, const int &b);
};

APSP::APSP(int32_t n) : num_vertex(n) {
    // Constructor, initialize AdjMatrix with 0 or MaxDistance
    if (n < 16) {
        block_size = n;
    }
    int res(n % block_size);
    if (res == 0) {
        n_oversized = n;
    } else {
        n_oversized = n + block_size - res;
    }
    AdjMatrix = new int[n_oversized * n_oversized]();
    for (int i = 0; i < n_oversized; ++i) {
        for (int j = 0; j < n_oversized; ++j) {
            if(i >= n || j >= n) {
                AdjMatrix[i * n_oversized +j] = MaxDistance;
            }
        }
    }
}

void APSP::FloydWarshall(const int &n, const int &b) {
    // for now, assume b divides n
    const int blocks = n / b;

    // note that [i][j] == [i * input_width * block_width + j * block_width]
    for (int k = 0; k < blocks; k++) {
    ispc::floyd_warshall_in_place(&AdjMatrix[k*b*n + k*b], &AdjMatrix[k*b*n + k*b], &AdjMatrix[k*b*n + k*b], b, n);
#pragma omp parallel for
    for (int j = 0; j < blocks; j++) {
        if (j == k) continue;
        ispc::floyd_warshall_in_place(&AdjMatrix[k*b*n + j*b], &AdjMatrix[k*b*n + k*b], &AdjMatrix[k*b*n + j*b], b, n);
    }
#pragma omp parallel for
    for (int i = 0; i < blocks; i++) {
        if (i == k) continue;
        ispc::floyd_warshall_in_place(&AdjMatrix[i*b*n + k*b], &AdjMatrix[i*b*n + k*b], &AdjMatrix[k*b*n + k*b], b, n);
        for (int j = 0; j < blocks; j++) {
            if (j == k) continue;
            ispc::floyd_warshall_in_place(&AdjMatrix[i*b*n + j*b], &AdjMatrix[i*b*n + k*b], &AdjMatrix[k*b*n + j*b], b, n);
        }
    }
  }
}

inline void APSP::AddEdge(int32_t from, int32_t to, int32_t weight) {
    size_t idx(from * n_oversized + to);
    AdjMatrix[idx] = weight;
}

void APSP::PrintData() {
    for (int i = 0; i < n_oversized; ++i) {
        for (int j = 0; j < n_oversized; ++j) {
            std::cout << std::setw(6) << AdjMatrix[i * n_oversized + j];
        }
        std::cout << std::endl;
    }
}

void APSP::Write_file(const char *filename) {
    std::ofstream out_file(filename);
    for (int32_t i = 0; i < num_vertex; i++) {
        for (int32_t j = 0; j < num_vertex; j++) {
            if (AdjMatrix[i * n_oversized + j] == INT16_MAX) {
                // Write 2^30 -1
                out_file.write((char *)&Out_max, sizeof(Out_max));
            } else {
                // Write normal
                out_file.write((char *)&AdjMatrix[i * n_oversized + j], sizeof(int32_t));
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
    int n_blocked = num_vertex;
    int block_remainder = num_vertex % block_size;
    if (block_remainder != 0) {
        n_blocked = num_vertex + block_size - block_remainder;
    }
    apsp.PrintData();
    apsp.FloydWarshall(n_blocked, block_size);
    apsp.PrintData();
    apsp.Write_file(argv[2]);
    return 0;
}