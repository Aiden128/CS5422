//#include "floyd_warshall.h"
#include "tbb/tbb.h"
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>

const int32_t MaxDistance = INT16_MAX;
const int32_t Out_max = 1073741823;

class APSP {
  private:
    int32_t num_vertex;
    std::vector<std::vector<int32_t>> AdjMatrix;

  public:
    APSP() : num_vertex(0){};
    APSP(int32_t n);
    inline void AddEdge(int32_t from, int32_t to, int32_t weight);
    void PrintData(std::vector<std::vector<int32_t>> array);
    void Write_file(const char *filename);
    void FloydWarshall();
    void Parallel_FloydWarshall();
};

APSP::APSP(int32_t n) : num_vertex(n) {
    // Constructor, initialize AdjMatrix with 0 or MaxDistance
    AdjMatrix.resize(num_vertex);
#pragma omp parallel for 
    for (int32_t i = 0; i < num_vertex; i++) {
        AdjMatrix[i].resize(num_vertex, MaxDistance);
        for (int32_t j = 0; j < num_vertex; j++) {
            if (i == j) {
                AdjMatrix[i][j] = 0;
            }
        }
    }
}

void APSP::FloydWarshall() {
    for (int32_t k = 0; k < num_vertex; k++) {
        for (int32_t i = 0; i < num_vertex; i++) {
            for (int32_t j = 0; j < num_vertex; j++) {
                AdjMatrix[i][j] = std::min(AdjMatrix[i][j], (AdjMatrix[i][k] + AdjMatrix[k][j]));
            }
        }
    }
}

void APSP::Parallel_FloydWarshall() {
    int32_t i, j, k;

#pragma omp parallel shared(num_vertex, AdjMatrix) private(i, j, k) default(none)
    for (k = 0; k < num_vertex; k++) {
#pragma omp for schedule(dynamic)
        for (i = 0; i < num_vertex; i++) {
            for (j = 0; j < num_vertex; j++) {
                AdjMatrix[i][j] = std::min(AdjMatrix[i][j], (AdjMatrix[i][k] + AdjMatrix[k][j]));
            }
        }
    }
}

void APSP::PrintData(std::vector<std::vector<int32_t>> array) {
    for (int32_t i = 0; i < num_vertex; i++) {
        for (int32_t j = 0; j < num_vertex; j++) {
            std::cout << std::setw(5) << array[i][j];
        }
        std::cout << std::endl;
    }
}

inline void APSP::AddEdge(int32_t from, int32_t to, int32_t weight) {
    AdjMatrix[from][to] = weight;
}

void APSP::Write_file(const char *filename) {
    std::ofstream out_file(filename);
    for (int32_t i = 0; i < num_vertex; i++) {
        for (int32_t j = 0; j < num_vertex; j++) {
            if (AdjMatrix[i][j] == INT16_MAX) {
                // Write 2^30 -1
                out_file.write((char *)&Out_max, sizeof(Out_max));
            } else {
                // Write normal
                out_file.write((char *)&AdjMatrix[i][j], sizeof(int32_t));
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
    for (int32_t i = 0; i < num_edge; ++i) {
        file.read((char *)&src, sizeof(src));
        file.read((char *)&dest, sizeof(dest));
        file.read((char *)&weight, sizeof(weight));
        apsp.AddEdge(src, dest, weight);
    }
    file.close();

    //apsp.FloydWarshall();
    apsp.Parallel_FloydWarshall();
    apsp.Write_file(argv[2]);
    return 0;
}