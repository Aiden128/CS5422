#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include "utils.h"

const int X = 5;
const int Y = 5;
const int Z = 2;
const int SCALE = 8;

__device__ const int dev_filter[Z][X][Y] = { 
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0}, 
     {  2,  8, 12,  8,  2}, 
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1}, 
     { -4, -8,  0,  8,  4}, 
     { -6,-12,  0, 12,  6}, 
     { -4, -8,  0,  8,  4}, 
     { -1, -2,  0,  2,  1}} 
};

void __global__ sobel (unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels, size_t pitch) {
    int  x, y, i, v, u;
    int  R, G, B;
    double val[Z*3] = {0.0};
    double totalR = 0.0;
    double totalG = 0.0;
    double totalB = 0.0;
    int adjustX, adjustY, xBound, yBound;

    // Copy sobel mask to each thread
    __shared__ int mask[Z][X][Y];
    x = (blockIdx.x * blockDim.x + threadIdx.x) % Z;
    y = (blockIdx.x * blockDim.x + threadIdx.x) / Z % X;
    i = (blockIdx.x * blockDim.x + threadIdx.x) / Z / X % Y;
    mask[x][y][i] = dev_filter[x][y][i];
    __syncthreads();

    y = blockIdx.x * blockDim.x + threadIdx.x + 2;
    if (y >= height + 2) {
        return;
    }
    for (x = 2; x < width + 2; ++x) {
        for (i = 0; i < Z; ++i) {
            // inline function causes compile err...
            // Use adjust factor replace
            adjustX = (X % 2) ? 1 : 0;
            adjustY = (Y % 2) ? 1 : 0;
            xBound = X / 2;
            yBound = Y / 2;

            val[i * 3 + 2] = 0.0;
            val[i * 3 + 1] = 0.0;
            val[i * 3] = 0.0;
            for (v = -yBound; v < yBound + adjustY; ++v) {
                for (u = -xBound; u < xBound + adjustX; ++u) {
                        R = s[pitch * (y+v) + channels * (x + u) + 2];
                        G = s[pitch * (y+v) + channels * (x + u) + 1];
                        B = s[pitch * (y+v) + channels * (x + u) + 0];
                        val[i * 3 + 2] += R * mask[i][u + xBound][v + yBound];
                        val[i * 3 + 1] += G * mask[i][u + xBound][v + yBound];
                        val[i * 3 + 0] += B * mask[i][u + xBound][v + yBound];
                }
            }
        }
        totalR = 0.0;
        totalG = 0.0;
        totalB = 0.0;
        for (i = 0; i < Z; ++i) {
            totalR += val[i * 3 + 2] * val[i * 3 + 2];
            totalG += val[i * 3 + 1] * val[i * 3 + 1];
            totalB += val[i * 3 + 0] * val[i * 3 + 0];
        }
        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;
        t[pitch * y + channels * x + 2] = (totalR > 255.0) ? 255 : totalR;
        t[pitch * y + channels * x + 1] = (totalG > 255.0) ? 255 : totalG;
        t[pitch * y + channels * x + 0] = (totalB > 255.0) ? 255 : totalB;
    }
}

int main(int argc, char** argv) {

    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char* host_s(nullptr);
    unsigned char *device_s(nullptr);
    unsigned char *device_t(nullptr);
    size_t pitch;
    read_png(argv[1], &host_s, &height, &width, &channels);

    // Reserve 2 pixels for each edge to prevent size becomes smaller after conv
    cudaMallocPitch(&device_s, &pitch, (width+4) * sizeof(unsigned char)* channels, (height+4));
    cudaMallocPitch(&device_t, &pitch,(width+4) * sizeof(unsigned char)* channels, (height+4));
    unsigned char* host_t = (unsigned char*) malloc(height * width * channels * sizeof(unsigned char));

    cudaMemcpy2D(device_s+2*pitch+2*channels, pitch, host_s, width * sizeof(unsigned char)* channels ,width * channels * sizeof(unsigned char), height, cudaMemcpyHostToDevice);
    sobel<<<(height/256) + 1, 256>>>(device_s, device_t, height, width, channels,pitch);
    cudaMemcpy2D(host_t , width * sizeof(unsigned char)* channels, device_t+2*pitch+2*channels, pitch , width * channels * sizeof(unsigned char), height, cudaMemcpyDeviceToHost);

    write_png(argv[2], host_t, height, width, channels);

    cudaFree(device_s);
    cudaFree(device_t);
    free(host_s);
    free(host_t);

    return 0;
}
