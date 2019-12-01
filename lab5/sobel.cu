#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <cuda.h>
#include <cudnn.h>
#include "utils/utils.h"

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

const int kernel_size = 5;
const int tile_size = 12;
const int bound = 2;
const int block_size = tile_size + kernel_size - 1;

__device__ __constant__ int filter[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                        { -2, -8, -12, -8, -2 },
                                        { 0, 0, 0, 0, 0 },
                                        { 2, 8, 12, 8, 2 },
                                        { 1, 4, 6, 4, 1 } },
                                      { { -1, -2, 0, 2, 1 },
                                        { -4, -8, 0, 8, 4 },
                                        { -6, -12, 0, 12, 6 },
                                        { -4, -8, 0, 8, 4 },
                                        { -1, -2, 0, 2, 1 } } };

inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

__global__ void sobel_share(unsigned char *s, unsigned char *t, unsigned height, unsigned width) {
    __shared__ unsigned char tile_shared[block_size][block_size];
    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get the output indices
    int row_o = ty + blockIdx.y * tile_size;
    int col_o = tx + blockIdx.x * tile_size;

    // shift to obtain input indices
    int row_i = row_o - bound;
    int col_i = col_o - bound;

    // Load tile elements
    if(row_i >= 0 && row_i < height && col_i >= 0 && col_i < width) {
        tile_shared[ty][tx] = s[row_i * width + col_i];
    } else {
        tile_shared[ty][tx] = 0;
     }
    // Wait until all tile elements are loaded
    __syncthreads();

    // only compute if you're an output tile element
    if(tx < tile_size && ty < tile_size){
        float Value_0 = 0.0f;
        float Value_1 = 0.0f;
        for(int y=0; y<kernel_size; y++) {
            for(int x=0; x<kernel_size; x++) {
                Value_0 += filter[0][y][x] * tile_shared[y+ty][x+tx];
                Value_1 += filter[1][y][x] * tile_shared[y+ty][x+tx];
            }
        }
        // only write values if you are inside matrix bounds
        if(row_o < height && col_o < width) {
            float ans = sqrt(Value_0 * Value_0 + Value_1 * Value_1) / SCALE;
            t[row_o * width + col_o] = (ans > 255.) ? 255 : ans;
        }
    }
}

__global__ void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width) {
    //2D Index of current thread
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    double val[Z];

    if((x>=0 && x < width) && (y >=0 && y < height)) {
        for (int i = 0; i < Z; ++i) {
            val[i] = 0.;
            /* Y and X axis of filter */
            for (int v = -yBound; v <= yBound; ++v) {
                for (int u = -xBound; u <= xBound; ++u) {
                    if (__builtin_expect(bound_check(x + u, 0, width) && 
                                        bound_check(y + v, 0, height), true)) {
                        const unsigned char pix = s[(width * (y + v) + (x + u))];
                        val[i] += pix * filter[i][u + xBound][v + yBound];
                    }
                }
            }
        }
        double ans = sqrt(val[0] * val[0] + val[1] * val[1]) / SCALE;
        t[(width * y + x)] = (ans > 255.) ? 255 : ans;
    }
}

__global__ void split(unsigned char *src, unsigned char *R, unsigned char *G, unsigned char *B, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= width * height) return;
    else {
        R[tid] = src[3 * tid + 2];
        G[tid] = src[3 * tid + 1];
        B[tid] = src[3 * tid];
    }
}

__global__ void merge(unsigned char *R, unsigned char *G, unsigned char *B, unsigned char *dest, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= width * height) return;
    else {
        dest[3 * tid + 2] = R[tid];
        dest[3 * tid + 1] = G[tid];
        dest[3 * tid + 0] = B[tid];
    }
}

int main(int argc, char **argv) {
    unsigned height, width, channels;
    unsigned char *src, *dst;
    unsigned char *dsrc, *ddst;
    unsigned char *R_d, *G_d, *B_d;
    unsigned char *R_ans, *G_ans, *B_ans;

    /* read the image to src, and get height, width, channels */
    if (read_png(argv[1], &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }
    dst = (unsigned char *)malloc(height * width * channels * sizeof(unsigned char));
    // cudaMalloc(...) for device src and device dst
    cudaMalloc(&dsrc, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&ddst, height * width * channels * sizeof(unsigned char));
    cudaMalloc(&R_d, height * width * sizeof(unsigned char));
    cudaMalloc(&G_d, height * width * sizeof(unsigned char));
    cudaMalloc(&B_d, height * width * sizeof(unsigned char));
    cudaMalloc(&R_ans, height * width * sizeof(unsigned char));
    cudaMalloc(&G_ans, height * width * sizeof(unsigned char));
    cudaMalloc(&B_ans, height * width * sizeof(unsigned char));

    // decide to use how many blocks and threads
    dim3 blk(block_size, block_size, 1);
    dim3 grid(ceil((float) width / tile_size), ceil((float) height / tile_size));
    const int nStreams = 3;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char),
    cudaMemcpyHostToDevice);
    // Split image to 3 channels
    split <<<ceil(((float)width * height) / 256), 256>>> (dsrc, R_d, G_d, B_d, width, height);
    cudaDeviceSynchronize();
    // launch cuda kernel
    sobel_share <<<grid, blk, 0, streams[0]>>> (R_d, R_ans, height, width);
    sobel_share <<<grid, blk, 0, streams[1]>>> (G_d, G_ans, height, width);
    sobel_share <<<grid, blk, 0, streams[2]>>> (B_d, B_ans, height, width);
    // merge layer data
    merge <<<ceil(((float)width * height) / 256), 256>>> (R_ans, G_ans, B_ans, ddst, width, height);
    // cudaMemcpy(...) copy result image to host
    cudaMemcpyAsync(dst, ddst, height * width * channels * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    write_png(argv[2], dst, height, width, channels);

    // Free memory
    free(src);
    free(dst);
    cudaFree(dsrc);
    cudaFree(ddst);
    cudaFree(R_d);
    cudaFree(G_d);
    cudaFree(B_d);
    cudaFree(R_ans);
    cudaFree(G_ans);
    cudaFree(B_ans);

    return 0;
}
