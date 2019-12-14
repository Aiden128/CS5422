#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <fstream>
#include <string>
#include <iostream>

const int BLOCK_SIZE = 32;
const int INF = 1073741823;

// Graph structure
struct Graph {
    unsigned int nvertex;         // number of vertex
    std::unique_ptr<int[]> graph; // graph matrix

    explicit Graph(int nvertex) : nvertex(nvertex) {
        graph = std::unique_ptr<int[]>(new int[nvertex * nvertex]());
    }
};

static __global__ void phase1(const int blockId, const size_t pitch,
                              const int nvertex, int *const graph) {
    __shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    const int v1(BLOCK_SIZE * blockId + idy);
    const int v2(BLOCK_SIZE * blockId + idx);
    const int tId(v1 * pitch + v2);
    int newPath(0);

    // Copy data to shared memory
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraph[idy][idx] = graph[tId];
    } else {
        cacheGraph[idy][idx] = INF;
    }
    __syncthreads();
    // Early stop unused thread to reduce sync cost
    if (v1 > nvertex || v2 > nvertex) {
        return;
    }

#pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        newPath = cacheGraph[idy][u] + cacheGraph[u][idx];
        if (newPath < cacheGraph[idy][idx]) {
            cacheGraph[idy][idx] = newPath;
        }
        __syncthreads();
    }
    graph[tId] = cacheGraph[idy][idx];
}

static __global__ void phase2(const int blockId, const size_t pitch,
                              const int nvertex, int *const graph) {
    if (blockIdx.x == blockId) {
        return;
    }
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    int v1(BLOCK_SIZE * blockId + idy);
    int v2(BLOCK_SIZE * blockId + idx);
    int currentPath(0);
    int tId(v1 * pitch + v2);
    int newPath(0);
    __shared__ int cacheGraphBase[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];

    // Copy data to shared memory
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraphBase[idy][idx] = graph[tId];
    } else {
        cacheGraphBase[idy][idx] = INF;
    }
    if (blockIdx.y == 0) {
        v2 = BLOCK_SIZE * blockIdx.x + idx;
    } else {
        v1 = BLOCK_SIZE * blockIdx.x + idy;
    }
    tId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        currentPath = graph[tId];
    } else {
        currentPath = INF;
    }
    cacheGraph[idy][idx] = currentPath;
    __syncthreads();
    // Early stop unused thread to reduce sync cost
    if (v1 > nvertex || v2 > nvertex) {
        return;
    }

    // Compute i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraphBase[idy][u] + cacheGraph[u][idx];
            if (newPath < currentPath) {
                currentPath = newPath;
            }
            cacheGraph[idy][idx] = currentPath;
            __syncthreads();
        }
    } else {
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];
            if (newPath < currentPath) {
                currentPath = newPath;
            }
            cacheGraph[idy][idx] = currentPath;
            __syncthreads();
        }
    }
    graph[tId] = currentPath;
}

static __global__ void phase3(const int blockId, const size_t pitch,
                              const int nvertex, int *const graph) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) {
        return;
    }
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    const int v1(blockDim.y * blockIdx.y + idy);
    const int v2(blockDim.x * blockIdx.x + idx);
    const int v1Row(BLOCK_SIZE * blockId + idy);
    const int v2Col(BLOCK_SIZE * blockId + idx);
    int tId(0);
    int currentPath(0);
    int newPath(0);
    __shared__ int cacheGraphBaseRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cacheGraphBaseCol[BLOCK_SIZE][BLOCK_SIZE];

    // Copy data to shared memory
    if (v1Row < nvertex && v2 < nvertex) {
        tId = v1Row * pitch + v2;
        cacheGraphBaseRow[idy][idx] = graph[tId];
    } else {
        cacheGraphBaseRow[idy][idx] = INF;
    }
    if (v1 < nvertex && v2Col < nvertex) {
        tId = v1 * pitch + v2Col;
        cacheGraphBaseCol[idy][idx] = graph[tId];
    } else {
        cacheGraphBaseCol[idy][idx] = INF;
    }
    __syncthreads();
    // Early stop unused thread to reduce sync cost
    if (v1 > nvertex || v2 > nvertex) {
        return;
    }

    tId = v1 * pitch + v2;
    currentPath = graph[tId];
#pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
        if (currentPath > newPath) {
            currentPath = newPath;
        }
    }
    graph[tId] = currentPath;
}

static size_t _cudaMoveMemoryToDevice(const std::unique_ptr<Graph> &dataHost,
                                      int **graphDevice) {
    const size_t height(dataHost->nvertex);
    const size_t width(height * sizeof(int));
    size_t pitch(0);

    // Allocate GPU memory
    cudaMallocPitch(graphDevice, &pitch, width, height);
    // Copy input from host memory to GPU memory
    cudaMemcpy2D(*graphDevice, pitch, dataHost->graph.get(), width, width,
                 height, cudaMemcpyHostToDevice);

    return pitch;
}

static void _cudaMoveMemoryToHost(int *graphDevice,
                                  const std::unique_ptr<Graph> &dataHost,
                                  const size_t &pitch) {
    const size_t height(dataHost->nvertex);
    const size_t width(height * sizeof(int));

    // Copy result to host memory from GPU memory
    cudaMemcpy2D(dataHost->graph.get(), width, graphDevice, pitch, width,
                 height, cudaMemcpyDeviceToHost);
    // Free GPU memory
    cudaFree(graphDevice);
}

void cudaBlockedFW(const std::unique_ptr<Graph> &dataHost) {
    const int nvertex(dataHost->nvertex);
    const int tile_size(std::ceil((float)nvertex / BLOCK_SIZE));
    const dim3 gridPhase1(1, 1);
    const dim3 gridPhase2(tile_size, 2);
    const dim3 gridPhase3(tile_size, tile_size);
    const dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    int *graphDevice(NULL);

    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice);
    for (int blockID = 0; blockID < tile_size; ++blockID) {
        // phase1
        phase1 <<<gridPhase1, dimBlockSize>>>
            (blockID, pitch / sizeof(int), nvertex, graphDevice);
        // phase2
        phase2 <<<gridPhase2, dimBlockSize>>>
            (blockID, pitch / sizeof(int), nvertex, graphDevice);
        // phase3
        phase3 <<<gridPhase3, dimBlockSize>>>
            (blockID, pitch / sizeof(int), nvertex, graphDevice);
    }
    _cudaMoveMemoryToHost(graphDevice, dataHost, pitch);
}

void Write_file(const std::string &filename,
                const std::unique_ptr<Graph> &data) {
    std::ofstream out_file(filename);
    for (int i = 0; i < data->nvertex; ++i) {
        out_file.write((char *)&data->graph[i * data->nvertex],
                       sizeof(int) * data->nvertex);
    }
    out_file.close();
}

int main(int argc, char **argv) {
    const std::string out_filename(argv[2]);
    std::ifstream file(argv[1]);
    int num_vertex(0);
    int num_edge(0);
    int src(0);
    int dest(0);
    int weight(0);

    // Read input params from file
    file.read((char *)&num_vertex, sizeof(num_vertex));
    file.read((char *)&num_edge, sizeof(num_edge));
    std::unique_ptr<Graph> AdjMatrix(new Graph(num_vertex));
    // Set initial values
    std::fill_n(AdjMatrix->graph.get(), num_vertex * num_vertex, INF);
    for (int i = 0; i < num_vertex; i++) {
        AdjMatrix->graph[i * num_vertex + i] = 0;
    }
    // Build graph
    std::unique_ptr<int[]> tmp(new int[num_edge * 3]);
    file.read((char *)tmp.get(), sizeof(int) * num_edge * 3);
    for (int i = 0; i < num_edge; ++i) {
        src = tmp[i * 3];
        dest = tmp[i * 3 + 1];
        weight = tmp[i * 3 + 2];
        int idx(src * num_vertex + dest);
        AdjMatrix->graph[idx] = weight;
    }
    file.close();
    // Run APSP algorithm
    cudaBlockedFW(AdjMatrix);
    // Write results
    Write_file(out_filename, AdjMatrix);

    return 0;
}