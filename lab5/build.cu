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
    //assert(argc == 3);
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
    //std::cout << "Width: " << width << " Height: " << height << " Channel: " << channels << std::endl;

    dst = (unsigned char *)malloc(height * width * channels *
                                  sizeof(unsigned char));
    //cudaMallocHost()
    //cudaMallocHost(&dst, height * width * channels * sizeof(unsigned char));
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
    dim3 blk(16, 32);
    dim3 grid(ceil((float) width / blk.x), ceil((float) height / blk.y));
    const int nStreams = 3;
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    //auto Tstart = std::chrono::high_resolution_clock::now();
    // cudaMemcpy(...) copy source image to device (filter matrix if necessary)
    cudaMemcpy(dsrc, src, height * width * channels * sizeof(unsigned char),
    cudaMemcpyHostToDevice);
    // Split image to 3 channels
    split <<<ceil(((float)width * height) / 256), 256>>> (dsrc, R_d, G_d, B_d, width, height);
    cudaDeviceSynchronize();
    // launch cuda kernel
    sobel <<<grid, blk, 0, streams[0]>>> (R_d, R_ans, height, width);
    sobel <<<grid, blk, 0, streams[1]>>> (G_d, G_ans, height, width);
    sobel <<<grid, blk, 0, streams[2]>>> (B_d, B_ans, height, width);
    //cudaDeviceSynchronize();
    // merge layer data
    merge <<<ceil(((float)width * height) / 256), 256>>> (R_ans, G_ans, B_ans, ddst, width, height);
    // cudaMemcpy(...) copy result image to host
    cudaMemcpyAsync(dst, ddst, height * width * channels * sizeof(unsigned char),
    cudaMemcpyDeviceToHost);
    //auto Tend = std::chrono::high_resolution_clock::now();
    //std::chrono::duration<double> duration = Tend - Tstart;
    //std::cout << "Measured Compute Time: " << duration.count() << "s" << std::endl;
    write_png(argv[2], dst, height, width, channels);
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
