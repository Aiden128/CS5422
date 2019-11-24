#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>
#include <omp.h>

const int32_t INF = 1073741823;

class APSP {
public:
    explicit APSP(const char *filename);
    void Write_file(const char *filename);
    void FloydWarshall();
    void Parallel_FloydWarshall();
    void OMP_FloydWarshall();

  private:
    inline void AddEdge(const int32_t &from, const int32_t &to, const int32_t &weight);
    int32_t num_vertex;
    int32_t num_edge;
    int32_t thread_count;
    std::vector<std::vector<int32_t>> AdjMatrix;
};

APSP::APSP(const char *filename) {
    thread_count = omp_get_max_threads();
    
    std::ifstream file(filename);
    int32_t src(0), dest(0), weight(0);
    file.read((char *)&num_vertex, sizeof(num_vertex));
    file.read((char *)&num_edge, sizeof(num_edge));

    int *tmp(new int[num_edge * 3]);
    file.read((char*)tmp, sizeof(int) * num_edge * 3);

    AdjMatrix.resize(num_vertex);
#pragma omp parallel for num_threads(2) schedule(dynamic)
    for (int32_t i = 0; i < num_vertex; ++i) {
        AdjMatrix[i].resize(num_vertex, INF);
        AdjMatrix[i][i] = 0;
    }

    for (int32_t i = 0; i < num_edge; ++i) {
        src = tmp[i*3];
        dest = tmp[i*3+1];
        weight = tmp[i*3+2];
        AddEdge(src, dest, weight);
    }
    file.close();
    delete[](tmp);
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

void APSP::OMP_FloydWarshall() {
    int grain(num_vertex / thread_count);
    for (int k = 0; k < num_vertex; k++) {
#pragma omp parallel num_threads(thread_count) 
{
        int tid = omp_get_thread_num();
        int start = grain * tid;
        int end = tid == thread_count - 1 ? num_vertex : grain * (tid + 1);
        for (int i = start; i < end; ++i) {
            for (int j = 0; j < num_vertex; ++j) {
                AdjMatrix[i][j] = std::min(AdjMatrix[i][j], (AdjMatrix[i][k] + AdjMatrix[k][j]));
            }
        }
}
    }
}

inline void APSP::AddEdge(const int32_t &from, const int32_t &to, const int32_t &weight) {
    AdjMatrix[from][to] = weight;
}

void APSP::Write_file(const char *filename) {
    std::ofstream out_file(filename);
    for (int32_t i = 0; i < num_vertex; ++i) {
        out_file.write((char*) &AdjMatrix[i][0], sizeof(int) * num_vertex);
    }
    out_file.close();
}

int main(int argc, char **argv) {
    APSP apsp(argv[1]);

    apsp.OMP_FloydWarshall();
    apsp.Write_file(argv[2]);
    return 0;
}