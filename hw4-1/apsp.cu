#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <fstream>
#include <string>
#include <iostream>
const int TILE_WIDTH = 32;
const int INF = 1073741823;

/* Default structure for graph */
struct graphAPSPTopology {
    unsigned int nvertex;         // number of vertex in graph
    std::unique_ptr<int[]> graph; // graph matrix

    /* Constructor for init fields */
    graphAPSPTopology(int nvertex) : nvertex(nvertex) {
        int size = nvertex * nvertex;
        graph = std::unique_ptr<int[]>(new int[size]());
    }
};

void Write_file(const std::string &filename,
                const std::unique_ptr<graphAPSPTopology> &data);

void basic_FW(int *matrix, int size);
__global__ void basic_run(int *matrix, int size, int k);

void blocked_FW(int *matrix, int size);
__forceinline__ __device__ void block_calc(int *C, int *A, int *B, int bj,
                                           int bi);
__global__ void phase1(int *matrix, int size, int stage);
__global__ void phase2(int *matrix, int size, int stage, int base);
__global__ void phase3(int *matrix, int size, int stage, int base);

int main(int argc, char **argv) {
    std::fstream file;
    int num_vertex = 0;
    int num_edge = 0;
    int src = 0;
    int dest = 0;
    int weight = 0;
    std::string in_filename(argv[1]);
    std::string out_filename(argv[2]);

    file.open(in_filename, std::ios::in | std::ios::binary);
    file.read((char *)&num_vertex, sizeof(num_vertex));
    file.read((char *)&num_edge, sizeof(num_edge));
    std::unique_ptr<graphAPSPTopology> AdjMatrix(
        new graphAPSPTopology(num_vertex));

    std::fill_n(AdjMatrix->graph.get(), num_vertex * num_vertex, INF);
    for (int i = 0; i < num_vertex; i++) {
        AdjMatrix->graph[i * num_vertex + i] = 0;
    }
    // std::cout << "Vertex: " << num_vertex << ", Edge: " << num_edge
    //           << std::endl;
    int *tmp(new int[num_edge * 3]);
    file.read((char *)tmp, sizeof(int) * num_edge * 3);
    for (int i = 0; i < num_edge; ++i) {
        src = tmp[i * 3];
        dest = tmp[i * 3 + 1];
        weight = tmp[i * 3 + 2];
        int idx(src * num_vertex + dest);
        AdjMatrix->graph[idx] = weight;
    }
    file.close();
    basic_FW(AdjMatrix->graph.get(), AdjMatrix->nvertex);
    // blocked_FW(AdjMatrix->graph.get(), AdjMatrix->nvertex);

    Write_file(out_filename, AdjMatrix);
    delete[](tmp);

    return 0;
}

void Write_file(const std::string &filename,
                const std::unique_ptr<graphAPSPTopology> &data) {
    std::ofstream out_file(filename);
    for (int i = 0; i < data->nvertex; ++i) {
        out_file.write((char *)&data->graph[i * data->nvertex],
                       sizeof(int) * data->nvertex);
    }
    out_file.close();
}

// Basic version of Floyd-Warshall
void basic_FW(int *matrix, int size) {
    // allocate memory
    int *matrixOnGPU;
    cudaMalloc((void **)&matrixOnGPU, sizeof(int) * size * size);
    cudaMemcpy(matrixOnGPU, matrix, sizeof(int) * size * size,
               cudaMemcpyHostToDevice);

    // dimension
    dim3 dimGrid(size, size, 1);

    // run kernel
    for (int k = 0; k < size; ++k) {
        basic_run << <dimGrid, 1>>> (matrixOnGPU, size, k);
    }
    // get result back
    cudaMemcpy(matrix, matrixOnGPU, sizeof(int) * size * size,
               cudaMemcpyDeviceToHost);
    cudaFree(matrixOnGPU);
}

__global__ void basic_run(int *matrix, int size, int k) {
    // compute indexes
    const int i = blockIdx.y;
    const int j = blockIdx.x;
    const int i0 = i * size + j;
    const int i1 = i * size + k;
    const int i2 = k * size + j;

    // read in dependent values
    const int i_j_value = matrix[i0];
    const int i_k_value = matrix[i1];
    const int k_j_value = matrix[i2];

    // calculate shortest path
    if (i_k_value != INF && k_j_value != INF) {
        int sum = i_k_value + k_j_value;
        if (i_j_value == INF || sum < i_j_value)
            matrix[i0] = sum;
    }
}
