#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifdef PERF
//#include "perf.hpp"
#endif
#include <chrono>
#include <complex>
#include <iostream>
#include <limits>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <cassert>
#include <boost/range/irange.hpp>
#include <mpi.h>
#include <omp.h>

int tile_size;
int data_size;
double left, right;
double lower, upper;
double dx, dy;
int width, height;
int image_size;
int iters;
int num_thread;
int rank, task_num;
int *image(nullptr);
int tile_size(100);
const int MANAGER_ID(0);
enum tag {RESULT, DATA, TERMINATE};

void manager();
void worker();
void mandel_omp(const int &start_idx = 0, const int &end_idx = image_size, int *result_ptr);
void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

int main(int argc, char **argv) {
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    image_size = width * height;
    image = new int[image_size];
    dx = static_cast<double> ((right - left) / width);
    dy = static_cast<double> ((upper - lower) / height);

    // MPI initialize
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    tile_size = task_num == 1 ? width : 20;
    data_size = tile_size * height + 1;

    if(rank == 0) {
        // Manager
        manager();
        // Note that write png can be done in rank 0
        write_png(filename, iters, width, height, image);
        delete[](image);
    } else {
        // Worker
        worker();
    }

    MPI_Finalize();
    return 0;
}

void manager() {
    // Single node
    if (task_num == 1) {
        // Draw image use normal OpenMP
        mandel_omp(0, image_size, image);
        return;
    // Multi-node
    } else {
        MPI_Status status;
        int active_nodes(1), schedule_idx(0);
        while(active_nodes < task_num && schedule_idx < width) {
            ++active_nodes;
            schedule_idx += tile_size;
            MPI_Send(&schedule_idx, 1, MPI::INT, active_nodes, tag::DATA, MPI_COMM_WORLD);
        }
        do {
            MPI_Recv(image, image_size, MPI::INT, MPI_ANY_SOURCE, tag::RESULT, MPI_COMM_WORLD, &status);
            --active_nodes;
            int sender_id (status.MPI_SOURCE);
            int col_idx = image[0];
            int *color(image + 1);
            if (schedule_idx < width) {
                MPI_Send(&schedule_idx, 1, MPI::INT, sender_id, tag::DATA, MPI_COMM_WORLD);
            } else {
                MPI_Send(&schedule_idx, 1, MPI::INT, sender_id, tag::TERMINATE, MPI_COMM_WORLD);
            }
        } while(active_nodes > 1);
    }
}

void worker() {
    MPI_Status status;
    int col_idx;
    MPI_Recv(&col_idx, 1, MPI::INT, MANAGER_ID, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    while(status.MPI_TAG == tag::DATA) {
        mandel_omp(col_idx, col_idx + tile_size, image);
        MPI_Send(image, data_size, MPI::INT, MANAGER_ID, tag::RESULT, MPI_COMM_WORLD);
        MPI_Recv(&col_idx, 1, MPI::INT, MANAGER_ID, tag::DATA, MPI_COMM_WORLD, &status);
    }
}

void mandel_omp(const int &start_idx, const int &end_idx, int *result_ptr) {
    #pragma omp parallel for schedule(dynamic, 100)
    for(int pixel_idx = start_idx; pixel_idx < end_idx; ++pixel_idx) {
        double y0((pixel_idx / width) * dy + lower), x0((pixel_idx % width) * dx + left);
        int repeats(0);
        double x(0.0), y(0.0), length_squared(0.0);
        while (repeats < iters && length_squared < 4) {
            double temp(x * x - y * y + x0);
            y = 2 * x * y + y0;
            x = temp;
            length_squared = x * x + y * y;
            ++repeats;
        }
        image[pixel_idx] = repeats;
    }
}

void write_png(const char *filename, int iters, int width, int height,
               const int *buffer) {
    FILE *fp = fopen(filename, "wb");
    assert(fp);
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    assert(png_ptr);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    assert(info_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
    #pragma unroll 2
    for (int y = 0; y < height; ++y) {
        std::fill(row, row+row_size, 0);
        for (int x = 0; x < width; ++x) {
            int p = buffer[(height - 1 - y) * width + x];
            png_bytep color = row + x * 3;
            if (p != iters) {
                if (p & 16) {
                    color[0] = 240;
                    color[1] = color[2] = p % 16 * 16;
                } else {
                    color[0] = p % 16 * 16;
                }
            }
        }
        png_write_row(png_ptr, row);
    }
    free(row);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}