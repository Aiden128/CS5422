#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <fstream>
#include <string>
#include <iostream>
#include <omp.h>
#include <cuda_profiler_api.h>

constexpr int BLOCK_SIZE = 32;
constexpr int INF = 1073741823;

// Graph structure
struct Graph {
    unsigned int nvertex;         // number of vertex
    unsigned int comp_nvertex;    // compensate vertex, v % BLOCK_SIZE == 0
    std::unique_ptr<int[]> graph; // graph matrix

    explicit Graph(int nvertex) : nvertex(nvertex) {
        comp_nvertex =
            nvertex + (BLOCK_SIZE - ((nvertex - 1) % BLOCK_SIZE + 1));
        graph = std::unique_ptr<int[]>(new int[comp_nvertex * comp_nvertex]());
    }
};

void cudaBlockedFW(const std::unique_ptr<Graph> &AdjMatrix);
__global__ void phase1(const int blockID, const int nvertex, int *graph);
__global__ void phase2(const int blockID, const int nvertex, int *graph);
__global__ void phase3(const int blockID, const int nvertex, const int offset,
                       int *graph);
void Write_file(const std::string &filename,
                const std::unique_ptr<Graph> &data);

int main(int argc, char **argv) {
    const std::string out_filename(argv[2]);
    std::ifstream file(argv[1]);
    int num_vertex(0), num_edge(0);
    int src(0);
    int dest(0);
    int weight(0);

    // Read input params from file
    file.read((char *)&num_vertex, sizeof(num_vertex));
    file.read((char *)&num_edge, sizeof(num_edge));
    std::unique_ptr<Graph> AdjMatrix(new Graph(num_vertex));
    // Set initial values
    std::fill_n(AdjMatrix->graph.get(),
                AdjMatrix->comp_nvertex * AdjMatrix->comp_nvertex, INF);
    for (int i = 0; i < AdjMatrix->comp_nvertex; i++) {
        AdjMatrix->graph[i * AdjMatrix->comp_nvertex + i] = 0;
    }
    // Build graph
    std::unique_ptr<int[]> tmp(new int[num_edge * 3]);
    file.read((char *)tmp.get(), sizeof(int) * num_edge * 3);
    for (int i = 0; i < num_edge; ++i) {
        src = tmp[i * 3];
        dest = tmp[i * 3 + 1];
        weight = tmp[i * 3 + 2];
        int idx(src * AdjMatrix->comp_nvertex + dest);
        AdjMatrix->graph[idx] = weight;
    }
    file.close();
    // Floyd Warshall Algorithm
    cudaBlockedFW(AdjMatrix);
    // Write results
    Write_file(out_filename, AdjMatrix);

    return 0;
}

void cudaBlockedFW(const std::unique_ptr<Graph> &AdjMatrix) {
    // Run APSP algorithm
    const int nvertex(AdjMatrix->comp_nvertex);
    const int orig_nvertex(AdjMatrix->nvertex);
    const int block_num(std::ceil((float)nvertex / BLOCK_SIZE));
    dim3 gridPhase1(1, 1);
    dim3 gridPhase2(2, block_num - 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    int *graph_d[2];
    const size_t graph_size(nvertex * nvertex * sizeof(int));

#pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);
        cudaMalloc((void **)&graph_d[thread_id], graph_size);
        // divide data
        int block_per_thd(block_num / 2);
        int y_offset(block_per_thd * thread_id);
        if (thread_id == 1) {
            block_per_thd += block_num % 2;
        }
        dim3 gridPhase3(block_per_thd, block_num);
        const size_t cp_amount(nvertex * BLOCK_SIZE * block_per_thd * sizeof(int));
        const size_t block_row_size(BLOCK_SIZE * nvertex * sizeof(int));
        const size_t cache_size(BLOCK_SIZE * BLOCK_SIZE * sizeof(int));
        cudaMemcpy(graph_d[thread_id] + y_offset * BLOCK_SIZE * nvertex,
                   AdjMatrix->graph.get() + y_offset * BLOCK_SIZE * nvertex,
                   cp_amount, cudaMemcpyHostToDevice);
        for (int blockID = 0; blockID < block_num; ++blockID) {
            if (__builtin_expect(blockID >= y_offset && blockID < (y_offset + block_per_thd), true)) {
                cudaMemcpy(AdjMatrix->graph.get() + blockID * BLOCK_SIZE * nvertex,
                           graph_d[thread_id] + blockID * BLOCK_SIZE * nvertex,
                           block_row_size, cudaMemcpyDeviceToHost);
            }
#pragma omp barrier
            cudaMemcpy(graph_d[thread_id] + blockID * BLOCK_SIZE * nvertex,
                       AdjMatrix->graph.get() + blockID * BLOCK_SIZE * nvertex,
                       block_row_size, cudaMemcpyHostToDevice);
            phase1 << <gridPhase1, dimBlockSize, cache_size>>>
                (blockID, nvertex, graph_d[thread_id]);
            phase2 << <gridPhase2, dimBlockSize, 3 * cache_size>>>
                (blockID, nvertex, graph_d[thread_id]);
            phase3 << <gridPhase3, dimBlockSize, 3 * cache_size>>>
                (blockID, nvertex, y_offset, graph_d[thread_id]);
        }
        cudaMemcpy(AdjMatrix->graph.get() + y_offset * BLOCK_SIZE * nvertex,
                   graph_d[thread_id] + y_offset * BLOCK_SIZE * nvertex,
                   block_row_size * block_per_thd, cudaMemcpyDeviceToHost);
    }
}

__global__ void phase1(const int block_ID, const int nvertex, int *graph) {
    const int i(threadIdx.y);
    const int j(threadIdx.x);
    const int offset(BLOCK_SIZE * block_ID);
    int newPath(0);
    extern __shared__ int cache[];

    cache[i * BLOCK_SIZE + j] = graph[(i + offset) * nvertex + (j + offset)];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        newPath = cache[i * BLOCK_SIZE + k] + cache[k * BLOCK_SIZE + j];
        if (cache[i * BLOCK_SIZE + j] > newPath) {
            cache[i * BLOCK_SIZE + j] = newPath;
        }
    }
    graph[(i + offset) * nvertex + (j + offset)] = cache[i * BLOCK_SIZE + j];
}

__global__ void phase2(const int block_ID, const int nvertex, int *graph) {
    const int total_round(nvertex / BLOCK_SIZE);
    const int i(threadIdx.y);
    const int j(threadIdx.x);
    const int i_offset(blockIdx.x == 1
                           ? BLOCK_SIZE *
                                 ((blockIdx.y + block_ID + 1) % total_round)
                           : BLOCK_SIZE * block_ID);
    const int j_offset(blockIdx.x == 1
                           ? BLOCK_SIZE * block_ID
                           : BLOCK_SIZE *
                                 ((blockIdx.y + block_ID + 1) % total_round));
    int newPath(0);
    extern __shared__ int cache[];

    cache[i * BLOCK_SIZE + j] =
        graph[(i + i_offset) * nvertex + (j + j_offset)];
    cache[(i + BLOCK_SIZE) * BLOCK_SIZE + j] =
        graph[(i + i_offset) * nvertex + j + block_ID * BLOCK_SIZE];
    cache[(i + 2 * BLOCK_SIZE) * BLOCK_SIZE + j] =
        graph[(i + block_ID * BLOCK_SIZE) * nvertex + (j + j_offset)];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        newPath = cache[(i + BLOCK_SIZE) * BLOCK_SIZE + k] +
                  cache[(k + 2 * BLOCK_SIZE) * BLOCK_SIZE + j];
        if (cache[i * BLOCK_SIZE + j] > newPath) {
            cache[i * BLOCK_SIZE + j] = newPath;
            if (block_ID == i_offset / BLOCK_SIZE) {
                cache[(i + 2 * BLOCK_SIZE) * BLOCK_SIZE + j] = newPath;
            }
            if (block_ID == j_offset / BLOCK_SIZE) {
                cache[(i + BLOCK_SIZE) * BLOCK_SIZE + j] = newPath;
            }
        }
    }
    graph[(i + i_offset) * nvertex + (j + j_offset)] =
        cache[i * BLOCK_SIZE + j];
}

__global__ void phase3(const int block_ID, const int nvertex, const int offset,
                       int *graph) {
    const int i(threadIdx.y);
    const int j(threadIdx.x);
    const int i_offset(BLOCK_SIZE * (blockIdx.x + offset));
    const int j_offset(BLOCK_SIZE * blockIdx.y);
    int newPath(0);
    extern __shared__ int cache[];

    cache[i * BLOCK_SIZE + j] =
        graph[(i + i_offset) * nvertex + (j + j_offset)];
    cache[(i + BLOCK_SIZE) * BLOCK_SIZE + j] =
        graph[(i + i_offset) * nvertex + j + block_ID * BLOCK_SIZE];
    cache[(i + 2 * BLOCK_SIZE) * BLOCK_SIZE + j] =
        graph[(i + block_ID * BLOCK_SIZE) * nvertex + (j + j_offset)];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        newPath = cache[(i + BLOCK_SIZE) * BLOCK_SIZE + k] +
                  cache[(k + 2 * BLOCK_SIZE) * BLOCK_SIZE + j];
        if (cache[i * BLOCK_SIZE + j] > newPath) {
            cache[i * BLOCK_SIZE + j] = newPath;
        }
    }
    graph[(i + i_offset) * nvertex + (j + j_offset)] =
        cache[i * BLOCK_SIZE + j];
}

void Write_file(const std::string &filename,
                const std::unique_ptr<Graph> &data) {
    std::ofstream out_file(filename);
    for (int i = 0; i < data->nvertex; ++i) {
        out_file.write((char *)&data->graph[i * data->comp_nvertex],
                       sizeof(int) * data->nvertex);
    }
    out_file.close();
}