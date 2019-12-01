#include <cstdlib>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iostream>
#include <random>
#include <iomanip>

using namespace std;

// __constant__ int filter[3][3] = {{1,2,3}, {4,5,6}, {7,8,9}};

// __global__ void conv(int* input, int* output, int width, int height) {
//     //2D Index of current thread
//     const int x = blockIdx.x * blockDim.x + threadIdx.x;
//     const int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if((x >= 0 && x < width) && (y >= 0 && y < height)) {

//     }
// }

__global__ void split(unsigned int *src, unsigned int *R, unsigned int *G, unsigned int *B, int width, int height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= width * height) return;
    R[tid] = src[3 * tid + 2];
    G[tid] = src[3 * tid + 1];
    B[tid] = src[3 * tid];
}

int main (void) {
    const int width(30), height(10);
    unsigned int *input(new unsigned int[width * height]());
    unsigned int *dev_input;
    unsigned int *R(new unsigned int[width*height/3]());
    unsigned int *G(new unsigned int[width*height/3]());
    unsigned int *B(new unsigned int[width*height/3]());
    unsigned int *R_d, *G_d, *B_d;
    cudaMalloc(&dev_input, height * width * sizeof(unsigned int));
    cudaMalloc(&R_d, height * width * sizeof(unsigned int) / 3);
    cudaMalloc(&G_d, height * width * sizeof(unsigned int) / 3);
    cudaMalloc(&B_d, height * width * sizeof(unsigned int) / 3);

    // Init inputs
    std::random_device rd;
    std::mt19937 generator(rd());
    std::uniform_int_distribution<int> distribution(0, 500);
    for (int i = 0; i < width * height; ++i) {
        input[i] = (distribution(generator));
    }
    cout << "Input Matrix" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            cout << setw(4) << input[i * width + j];
        }
        cout << endl;
    }

    cudaMemcpy(dev_input, input, height * width * sizeof(unsigned int), cudaMemcpyHostToDevice);
    split <<<ceil(((float)width * height / 3) / 32), 32>>> (dev_input, R_d, G_d, B_d, width/3, height);
    cudaMemcpy(R, R_d, height * width * sizeof(unsigned int) / 3,cudaMemcpyDeviceToHost);
    cudaMemcpy(G, G_d, height * width * sizeof(unsigned int) / 3,cudaMemcpyDeviceToHost);
    cudaMemcpy(B, B_d, height * width * sizeof(unsigned int) / 3,cudaMemcpyDeviceToHost);

    // Print results
    cout << "R" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width / 3; ++j) {
            cout << setw(4) << R[(i * width /3) + j];
        }
        cout << endl;
    }
    cout << "G" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width / 3; ++j) {
            cout << setw(4) << G[(i * width /3) + j];
        }
        cout << endl;
    }
    cout << "B" << endl;
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width / 3; ++j) {
            cout << setw(4) << B[(i * width /3) + j];
        }
        cout << endl;
    }

    cudaFree(dev_input);
    cudaFree(R_d);
    cudaFree(G_d);
    cudaFree(B_d);

    delete[](R);
    delete[](G);
    delete[](B);
    delete[](input);
    return 0;
}