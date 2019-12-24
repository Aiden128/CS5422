#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <fstream>
#include <string>
#include <iostream>
#include <cuda_profiler_api.h>

constexpr int BLOCK_SIZE = 32;
constexpr int INF = 1073741823;

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
    __shared__ int cache[BLOCK_SIZE][BLOCK_SIZE];
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    const int v1(BLOCK_SIZE * blockId + idy);
    const int v2(BLOCK_SIZE * blockId + idx);
    const int tId(v1 * pitch + v2);
    int newPath(0);

    // Copy data to shared memory
    if (v1 < nvertex && v2 < nvertex) {
        cache[idy][idx] = graph[tId];
    } else {
        cache[idy][idx] = INF;
    }
    __syncthreads();
    // Early stop unused thread to reduce sync cost
    if (v1 >= nvertex || v2 >= nvertex) {
        return;
    }

#pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        newPath = cache[idy][u] + cache[u][idx];
        if (newPath < cache[idy][idx]) {
            cache[idy][idx] = newPath;
        }
        __syncthreads();
    }
    graph[tId] = cache[idy][idx];
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
    __shared__ int cacheBase[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cache[BLOCK_SIZE][BLOCK_SIZE];

    // Copy data to shared memory
    if (v1 < nvertex && v2 < nvertex) {
        cacheBase[idy][idx] = graph[tId];
    } else {
        cacheBase[idy][idx] = INF;
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
    cache[idy][idx] = currentPath;
    __syncthreads();
    // Early stop unused thread to reduce sync cost
    if (v1 >= nvertex || v2 >= nvertex) {
        return;
    }

    if (blockIdx.y == 0) {
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheBase[idy][u] + cache[u][idx];
            if (newPath < currentPath) {
                currentPath = newPath;
            }
            cache[idy][idx] = currentPath;
        }
    } else {
#pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cache[idy][u] + cacheBase[u][idx];
            if (newPath < currentPath) {
                currentPath = newPath;
            }
            cache[idy][idx] = currentPath;
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
    const int tId(v1 * pitch + v2);
    int index(0);
    int currentPath(0);
    int newPath(0);
    __shared__ int cacheRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cacheCol[BLOCK_SIZE][BLOCK_SIZE];

    // Copy data to shared memory
    if (v1Row < nvertex && v2 < nvertex) {
        index = (v1Row * pitch + v2);
        cacheRow[idy][idx] = graph[index];
    } else {
        cacheRow[idy][idx] = INF;
    }
    if (v1 < nvertex && v2Col < nvertex) {
        index = (v1 * pitch + v2Col);
        cacheCol[idy][idx] = graph[index];
    } else {
        cacheCol[idy][idx] = INF;
    }
    __syncthreads();
    // Early stop unused thread to reduce sync cost
    if (v1 >= nvertex || v2 >= nvertex) {
        return;
    }

    currentPath = graph[tId];
#pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        newPath = cacheCol[idy][u] + cacheRow[u][idx];
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
    const int block_num(std::ceil((float)nvertex / BLOCK_SIZE));
    const dim3 gridPhase1(1, 1);
    const dim3 gridPhase2(block_num, 2);
    const dim3 gridPhase3(block_num, block_num);
    const dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    int *graphDevice(NULL);

    cudaProfilerStart();
    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice);
    for (int blockID = 0; blockID < block_num; ++blockID) {
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
    cudaProfilerStop();
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