#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>
#include <fstream>
#include <string>
#include <iostream>
const int BLOCK_SIZE = 32;
const int INF = 1073741823;

/* Default structure for graph */
struct graphAPSPTopology {
    unsigned int nvertex; // number of vertex in graph
    std::unique_ptr<int[]> graph; // graph matrix

    /* Constructor for init fields */
    graphAPSPTopology(int nvertex): nvertex(nvertex) {
        int size = nvertex * nvertex;
        graph = std::unique_ptr<int[]>(new int[size]());
    }
};

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Dependent phase 1
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 */
static __global__
void _blocked_fw_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph) {
    __shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    const int v1(BLOCK_SIZE * blockId + idy);
    const int v2(BLOCK_SIZE * blockId + idx);
    int newPath(0);
    const int cellId(v1 * pitch + v2);

    if (v1 < nvertex && v2 < nvertex) {
        cacheGraph[idy][idx] = graph[cellId];
    } else {
        cacheGraph[idy][idx] = INF;
    }
    // Synchronize to make sure the all value are loaded in block
    __syncthreads();
    if (v1 > nvertex || v2 > nvertex) {
        return;
    }

    #pragma unroll
    for (int u = 0; u < BLOCK_SIZE; ++u) {
        newPath = cacheGraph[idy][u] + cacheGraph[u][idx];
        if (newPath < cacheGraph[idy][idx]) {
            cacheGraph[idy][idx] = newPath;
        }
        // Synchronize to make sure that all value are current
        __syncthreads();
    }
    graph[cellId] = cacheGraph[idy][idx];
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Partial dependent phase 2
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 */
static __global__
void _blocked_fw_partial_dependent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph) {
    if (blockIdx.x == blockId) return;
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    int v1(BLOCK_SIZE * blockId + idy);
    int v2(BLOCK_SIZE * blockId + idx);
    __shared__ int cacheGraphBase[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cacheGraph[BLOCK_SIZE][BLOCK_SIZE];
    int currentPath(0);
    int cellId(v1 * pitch + v2);
    int newPath(0);

    // Load base block for graph and predecessors
    if (v1 < nvertex && v2 < nvertex) {
        cacheGraphBase[idy][idx] = graph[cellId];
    } else {
        cacheGraphBase[idy][idx] = INF;
    }

    // Load i-aligned singly dependent blocks
    if (blockIdx.y == 0) {
        v2 = BLOCK_SIZE * blockIdx.x + idx;
    } else {
    // Load j-aligned singly dependent blocks
        v1 = BLOCK_SIZE * blockIdx.x + idy;
    }

    // Load current block for graph and predecessors
    cellId = v1 * pitch + v2;
    if (v1 < nvertex && v2 < nvertex) {
        currentPath = graph[cellId];
    } else {
        currentPath = INF;
    }
    cacheGraph[idy][idx] = currentPath;
    // Synchronize to make sure the all value are saved in cache
    __syncthreads();

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
            // Update new values
            cacheGraph[idy][idx] = currentPath;
            // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    } else {
    // Compute j-aligned singly dependent blocks
        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
            newPath = cacheGraph[idy][u] + cacheGraphBase[u][idx];
            if (newPath < currentPath) {
                currentPath = newPath;
            }
            // Update new values
            cacheGraph[idy][idx] = currentPath;
            // Synchronize to make sure that all threads update cache
            __syncthreads();
        }
    }
    graph[cellId] = currentPath;
}

/**
 * Blocked CUDA kernel implementation algorithm Floyd Wharshall for APSP
 * Independent phase 3
 *
 * @param blockId: Index of block
 * @param nvertex: Number of all vertex in graph
 * @param pitch: Length of row in memory
 * @param graph: Array of graph with distance between vertex on device
 */
static __global__
void _blocked_fw_independent_ph(const int blockId, size_t pitch, const int nvertex, int* const graph) {
    if (blockIdx.x == blockId || blockIdx.y == blockId) return;
    const int idx(threadIdx.x);
    const int idy(threadIdx.y);
    const int v1(blockDim.y * blockIdx.y + idy);
    const int v2(blockDim.x * blockIdx.x + idx);
    __shared__ int cacheGraphBaseRow[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int cacheGraphBaseCol[BLOCK_SIZE][BLOCK_SIZE];
    int v1Row(BLOCK_SIZE * blockId + idy);
    int v2Col(BLOCK_SIZE * blockId + idx);
    int cellId(0);
    int currentPath(0);
    int newPath(0);

    // Load data for block
    if (v1Row < nvertex && v2 < nvertex) {
        cellId = v1Row * pitch + v2;
        cacheGraphBaseRow[idy][idx] = graph[cellId];
    }
    else {
        cacheGraphBaseRow[idy][idx] = INF;
    }
    if (v1  < nvertex && v2Col < nvertex) {
        cellId = v1 * pitch + v2Col;
        cacheGraphBaseCol[idy][idx] = graph[cellId];
    }
    else {
        cacheGraphBaseCol[idy][idx] = INF;
    }
    // Synchronize to make sure the all value are loaded in virtual block
    __syncthreads();

    if (v1 > nvertex || v2 > nvertex) {
        return;
    }

    // Compute data for block
    if (v1  < nvertex && v2 < nvertex) {
       cellId = v1 * pitch + v2;
       currentPath = graph[cellId];

        #pragma unroll
        for (int u = 0; u < BLOCK_SIZE; ++u) {
           newPath = cacheGraphBaseCol[idy][u] + cacheGraphBaseRow[u][idx];
           if (currentPath > newPath) {
               currentPath = newPath;
           }
       }
       graph[cellId] = currentPath;
   }
}

/**
 * Allocate memory on device and copy memory from host to device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param graphDevice: Pointer to array of graph with distance between vertex on device
 *
 * @return: Pitch for allocation
 */
static
size_t _cudaMoveMemoryToDevice(const std::unique_ptr<graphAPSPTopology>& dataHost, int **graphDevice) {
    size_t height = dataHost->nvertex;
    size_t width = height * sizeof(int);
    size_t pitch;

    // Allocate GPU buffers for matrix of shortest paths d(G) and predecessors p(G)
    cudaMallocPitch(graphDevice, &pitch, width, height);

    // Copy input from host memory to GPU buffers and
    cudaMemcpy2D(*graphDevice, pitch, dataHost->graph.get(), width, width, height, cudaMemcpyHostToDevice);

    return pitch;
}

/**
 * Copy memory from device to host and free device memory
 *
 * @param graphDevice: Array of graph with distance between vertex on device
 * @param dataHost: Reference to unique ptr to graph data with allocated fields on host
 * @param pitch: Pitch for allocation
 */
static
void _cudaMoveMemoryToHost(int *graphDevice, const std::unique_ptr<graphAPSPTopology>& dataHost, size_t pitch) {
    size_t height = dataHost->nvertex;
    size_t width = height * sizeof(int);

    cudaMemcpy2D(dataHost->graph.get(), width, graphDevice, pitch, width, height, cudaMemcpyDeviceToHost);
    cudaFree(graphDevice);
}

/**
 * Blocked implementation of Floyd Warshall algorithm in CUDA
 *
 * @param data: unique ptr to graph data with allocated fields on host
 */
void cudaBlockedFW(const std::unique_ptr<graphAPSPTopology>& dataHost) {
    // HANDLE_ERROR(cudaSetDevice(0));
    int nvertex(dataHost->nvertex);
    int *graphDevice(NULL);
    size_t pitch = _cudaMoveMemoryToDevice(dataHost, &graphDevice);

    dim3 gridPhase1(1 ,1, 1);
    dim3 gridPhase2((nvertex - 1) / BLOCK_SIZE + 1, 2 , 1);
    dim3 gridPhase3((nvertex - 1) / BLOCK_SIZE + 1, (nvertex - 1) / BLOCK_SIZE + 1 , 1);
    dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    int numBlock = (nvertex - 1) / BLOCK_SIZE + 1;

    for(int blockID = 0; blockID < numBlock; ++blockID) {
        // Start dependent phase
        _blocked_fw_dependent_ph<<<gridPhase1, dimBlockSize>>>
                (blockID, pitch / sizeof(int), nvertex, graphDevice);

        // Start partially dependent phase
        _blocked_fw_partial_dependent_ph<<<gridPhase2, dimBlockSize>>>
                (blockID, pitch / sizeof(int), nvertex, graphDevice);

        // Start independent phase
        _blocked_fw_independent_ph<<<gridPhase3, dimBlockSize>>>
                (blockID, pitch / sizeof(int), nvertex, graphDevice);
    }

    _cudaMoveMemoryToHost(graphDevice, dataHost, pitch);
}

/**
 * Print data graph (graph matrix, prep) and time
 *
 * @param graph: pointer to graph data
 * @param max: maximum value in graph path
 */
 void printData(const std::unique_ptr<graphAPSPTopology>& graph, int maxValue) {
    // Lambda function for printMatrix -1 means no path
    std::ios::sync_with_stdio(false);
    auto printMatrix = [](std::unique_ptr<int []>& graph, int n, int max) {
        std::cout << "[";
        for (int i = 0; i < n; ++i) {
            std::cout << "[";
            for (int j = 0; j < n; ++j) {
                if (max > graph[i * n + j])
                    std::cout << graph[i * n + j];
                else
                    std::cout << -1 ;
                if (j != n - 1) std::cout << ",";
            }
            if (i != n - 1)
                std::cout << "],\n";
            else
                std::cout << "]";
        }
        std::cout << "],\n";
    };

    std::cout << "{\n    \"graph\":\n";
    printMatrix(graph->graph, graph->nvertex, maxValue);
}

void Write_file(const std::string &filename, const std::unique_ptr<graphAPSPTopology> &data) {
    std::ofstream out_file(filename);
    for (int i = 0; i < data->nvertex; ++i) {
        out_file.write((char *)&data->graph[i * data->nvertex], sizeof(int) * data->nvertex);
    }
    out_file.close();
}

int main (int argc, char **argv) {
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
    std::unique_ptr<graphAPSPTopology> AdjMatrix(new graphAPSPTopology(num_vertex));

    std::fill_n(AdjMatrix->graph.get(), num_vertex * num_vertex, INF);  
    for (int i = 0; i < num_vertex; i++) {
        AdjMatrix->graph[i * num_vertex + i] = 0;
    }

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
    cudaBlockedFW(AdjMatrix);
    Write_file(out_filename, AdjMatrix);
    delete[] (tmp);

    return 0;
}