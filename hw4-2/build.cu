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

void cudaBlockedFW(const std::unique_ptr<Graph> &dataHost);

__global__ void phase1(const int blockId, const int nvertex, int *graph);
__global__ void phase2(const int blockId, const int nvertex, int *graph);
__global__ void phase3(const int blockId, const int nvertex,
                       const int offset, int *graph);

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
    // Run APSP algorithm
    int comp_V = num_vertex + (BLOCK_SIZE - ((num_vertex-1) % BLOCK_SIZE + 1));
    int *adj_mat_d[2];
    int round = std::ceil((float) comp_V / BLOCK_SIZE);
    
    // 2D block
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 p1(1, 1);
    dim3 p2(2, round-1);
    size_t sz = comp_V * comp_V * sizeof(int);
    #pragma omp parallel num_threads(2)
    {
        int thread_id = omp_get_thread_num();
        cudaSetDevice(thread_id);

        // Malloc memory
        cudaMalloc((void**) &adj_mat_d[thread_id], sz);

        // divide data
        int round_per_thd = round / 2;
        int y_offset = round_per_thd * thread_id;
        if(thread_id == 1)
            round_per_thd += round % 2;

        dim3 p3(round_per_thd, round);
        
        size_t cp_amount = comp_V * BLOCK_SIZE * round_per_thd * sizeof(int);
        cudaMemcpy(adj_mat_d[thread_id] + y_offset *BLOCK_SIZE * comp_V, AdjMatrix->graph.get() + y_offset * BLOCK_SIZE * comp_V, cp_amount, cudaMemcpyHostToDevice);

        size_t block_row_sz = BLOCK_SIZE * comp_V * sizeof(int);
        for(int r = 0; r < round; r++){    
            if (r >= y_offset && r < (y_offset + round_per_thd)) {
                cudaMemcpy(AdjMatrix->graph.get() + r * BLOCK_SIZE * comp_V, adj_mat_d[thread_id] + r * BLOCK_SIZE * comp_V, block_row_sz, cudaMemcpyDeviceToHost);
            }
            #pragma omp barrier
            cudaMemcpy(adj_mat_d[thread_id] + r * BLOCK_SIZE * comp_V, AdjMatrix->graph.get() + r * BLOCK_SIZE * comp_V, block_row_sz, cudaMemcpyHostToDevice);

            phase1 <<<p1, threads, sizeof(int)*BLOCK_SIZE*BLOCK_SIZE >>>(r, comp_V, adj_mat_d[thread_id]);
            
            cudaDeviceSynchronize();
            
            phase2 <<<p2, threads, sizeof(int)*3*BLOCK_SIZE*BLOCK_SIZE >>>(r, comp_V, adj_mat_d[thread_id]);
            
            cudaDeviceSynchronize();
            
            phase3 <<<p3, threads, sizeof(int)*3*BLOCK_SIZE*BLOCK_SIZE >>>(r, comp_V, y_offset, adj_mat_d[thread_id]);
        }
        cudaMemcpy(AdjMatrix->graph.get() + y_offset *BLOCK_SIZE * comp_V, adj_mat_d[thread_id] + y_offset *BLOCK_SIZE * comp_V, block_row_sz * round_per_thd, cudaMemcpyDeviceToHost);
        #pragma omp barrier
    }

    // Write results
    Write_file(out_filename, AdjMatrix);

    return 0;
}

void cudaBlockedFW(const std::unique_ptr<Graph> &dataHost) {
    const int nvertex(dataHost->comp_nvertex);
    const int block_num(std::ceil((float)nvertex / BLOCK_SIZE));
    const dim3 gridPhase1(1, 1);
    const dim3 gridPhase2(block_num, 2);
    const dim3 dimBlockSize(BLOCK_SIZE, BLOCK_SIZE);
    const size_t size(nvertex * nvertex * sizeof(int));
    int *graphDevice[2];

#pragma omp parallel num_threads(2)
    {
        int thread_id(omp_get_thread_num());
        cudaSetDevice(thread_id);
        cudaMalloc((void **)&graphDevice[thread_id], size);
        // Divide data
        int block_per_thread(block_num / 2);
        if (thread_id == 1) {
            block_per_thread += block_num % 2;
        }
        const dim3 gridPhase3(block_per_thread, block_num);
        const int y_offset = block_per_thread * thread_id; // offset of y axis
        const size_t copy_size(nvertex * BLOCK_SIZE * block_per_thread *
                               sizeof(int));
        const int offset(y_offset * BLOCK_SIZE * nvertex);
        cudaMemcpy(graphDevice[thread_id] + offset,
                   dataHost->graph.get() + offset, copy_size,
                   cudaMemcpyHostToDevice);
        const size_t block_row_size(BLOCK_SIZE * nvertex * sizeof(int));
        const size_t shared_size(BLOCK_SIZE * BLOCK_SIZE * sizeof(int));
        for (int blockID = 0; blockID < block_num; ++blockID) {
            if ((blockID >= y_offset) &&
                (blockID < (y_offset + block_per_thread))) {
                cudaMemcpy(
                    dataHost->graph.get() + blockID * BLOCK_SIZE * nvertex,
                    graphDevice[thread_id] + blockID * BLOCK_SIZE * nvertex,
                    block_row_size, cudaMemcpyDeviceToHost);
            }
#pragma omp barrier
            cudaMemcpy(graphDevice[thread_id] + blockID * BLOCK_SIZE * nvertex,
                       dataHost->graph.get() + blockID * BLOCK_SIZE * nvertex,
                       block_row_size, cudaMemcpyHostToDevice);

            phase1 << <gridPhase1, dimBlockSize, shared_size>>>
                (blockID, nvertex, graphDevice[thread_id]);
            cudaDeviceSynchronize();
            phase2 << <gridPhase2, dimBlockSize, 3 * shared_size>>>
                (blockID, nvertex, graphDevice[thread_id]);
            cudaDeviceSynchronize();
            phase3 << <gridPhase3, dimBlockSize, 3 * shared_size>>>
                (blockID, nvertex, y_offset, graphDevice[thread_id]);
        }
        cudaMemcpy(dataHost->graph.get() + offset,
                   graphDevice[thread_id] + offset, copy_size,
                   cudaMemcpyDeviceToHost);
#pragma omp barrier
    }
}

// phase 1 kernel
__global__ void phase1(const int round, const int comp_V, int *adj_mat_d) {
    
    int i = threadIdx.y, 
        j = threadIdx.x,
        offset = BLOCK_SIZE * round;
    
    extern __shared__ int shared_mem[];

    shared_mem[i * BLOCK_SIZE + j] = adj_mat_d[(i + offset) * comp_V + (j + offset)];
    __syncthreads();

#pragma unroll
    for(int k = 0; k < BLOCK_SIZE; k++){
        if (shared_mem[i * BLOCK_SIZE + j] > shared_mem[i * BLOCK_SIZE + k] + shared_mem[k * BLOCK_SIZE + j]){
            shared_mem[i * BLOCK_SIZE + j] = shared_mem[i * BLOCK_SIZE + k] + shared_mem[k * BLOCK_SIZE + j];
        }
    }
    adj_mat_d[(i + offset) * comp_V + (j + offset)] = shared_mem[i * BLOCK_SIZE + j];
}

// phase 2 kernel
__global__ void phase2(const int round, const int comp_V, int* adj_mat_d) {
    int total_round = comp_V/BLOCK_SIZE;
    int i = threadIdx.y,
        j = threadIdx.x,
        // column or row?
        i_off = blockIdx.x == 1? BLOCK_SIZE * ((blockIdx.y + round + 1) % total_round): BLOCK_SIZE * round,
        j_off = blockIdx.x == 1? BLOCK_SIZE * round : BLOCK_SIZE * ((blockIdx.y + round + 1) % total_round);
    
    extern __shared__ int shared_mem[];
    
    shared_mem[i * BLOCK_SIZE + j] = adj_mat_d[(i + i_off) * comp_V + (j+j_off)];
    shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + j] = adj_mat_d[(i + i_off) * comp_V + j + round*BLOCK_SIZE];
    shared_mem[(i + 2*BLOCK_SIZE) * BLOCK_SIZE + j] = adj_mat_d[(i + round * BLOCK_SIZE) * comp_V + (j + j_off)];
    __syncthreads();

#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (shared_mem[i * BLOCK_SIZE + j] > shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + k] + shared_mem[(k + 2*BLOCK_SIZE) * BLOCK_SIZE + j]) {
            shared_mem[i * BLOCK_SIZE + j] = shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + k] + shared_mem[(k + 2*BLOCK_SIZE) * BLOCK_SIZE + j]; 
            
            if (round == i_off/BLOCK_SIZE) 
                shared_mem[(i + 2*BLOCK_SIZE) * BLOCK_SIZE + j] = shared_mem[i * BLOCK_SIZE + j];
            if (round == j_off/BLOCK_SIZE) 
                shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + j] = shared_mem[i * BLOCK_SIZE + j];
        }
    }
    adj_mat_d[(i + i_off) * comp_V + (j+j_off)] = shared_mem[i * BLOCK_SIZE + j];
}

__global__ void phase3(const int round, const int comp_V, const int offset, int* adj_mat_d) {
    int i = threadIdx.y,
        j = threadIdx.x,
        i_off = BLOCK_SIZE * (blockIdx.x + offset),
        j_off = BLOCK_SIZE * blockIdx.y;
    extern __shared__ int shared_mem[];

    shared_mem[i * BLOCK_SIZE + j] = adj_mat_d[(i + i_off) * comp_V + (j+j_off)];
    shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + j] = adj_mat_d[(i + i_off) * comp_V + j + round*BLOCK_SIZE];
    shared_mem[(i + 2*BLOCK_SIZE) * BLOCK_SIZE + j] = adj_mat_d[(i + round * BLOCK_SIZE) * comp_V + (j + j_off)];
    __syncthreads();
    
#pragma unroll
    for (int k = 0; k < BLOCK_SIZE; k++) {
        if (shared_mem[i * BLOCK_SIZE + j] > shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + k] + shared_mem[(k + 2*BLOCK_SIZE) * BLOCK_SIZE + j])
            shared_mem[i * BLOCK_SIZE + j] = shared_mem[(i + BLOCK_SIZE) * BLOCK_SIZE + k] + shared_mem[(k + 2*BLOCK_SIZE) * BLOCK_SIZE + j];
    }
    adj_mat_d[(i + i_off) * comp_V + (j+j_off)] = shared_mem[i * BLOCK_SIZE + j];
}

void Write_file(const std::string &filename,
                const std::unique_ptr<Graph> &data) {
    std::ofstream out_file(filename);
    for (int i = 0; i < data->nvertex; ++i) {
        for (int j = 0; j < data->nvertex; ++j) {
            out_file.write((char *)&data->graph[i * data->comp_nvertex + j],
                           sizeof(int));
        }
    }
    out_file.close();
}