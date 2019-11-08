#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#ifdef PERF
//#include "perf.hpp"
#endif
#include <algorithm>
#include <boost/range/irange.hpp>
#include <cassert>
#include <chrono>
#include <complex>
#include <iostream>
#include <iterator>
#include <limits>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <pthread.h>
#include <sched.h>

enum tag { RESULT, DATA, TERMINATE };

void manager(int *image);
void worker();
void mandel_omp(const int &start_idx, int *result_ptr);
void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

int main(int argc, char **argv) {
    const char *filename(argv[1]);
    const int iters = strtol(argv[2], 0, 10);
    const double left = strtod(argv[3], 0);
    const double right = strtod(argv[4], 0);
    const double lower = strtod(argv[5], 0);
    const double upper = strtod(argv[6], 0);
    const int width = strtol(argv[7], 0, 10);
    const int height = strtol(argv[8], 0, 10);
    const double dx = static_cast<double>((right - left) / width);
    const double dy = static_cast<double>((upper - lower) / height);
    const int image_size(width * height);
    const int tile(2);
    const int buffer_size(width * tile + 1);

    int task_num, rank;
    MPI_Status status;
    MPI_Request request;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (__builtin_expect(rank == 0, false)) {
        int *image(new int[image_size]);
        int *buffer(new int[buffer_size]);
        // Manager
        if (__builtin_expect(task_num == 1, false)) {
#pragma omp parallel for schedule(dynamic, 100)
            // Finish drawing smoothly
            for (int pixel_idx = 0; pixel_idx < image_size; ++pixel_idx) {
                double y0((pixel_idx / width) * dy + lower),
                    x0((pixel_idx % width) * dx + left);
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
        } else {
            int active_nodes(0);
            int node_id(1);
            for (int i = 0; i < height; i += tile) {
                if (node_id < task_num) {
                    MPI_Isend(&i, 1, MPI::INT, node_id, tag::DATA,
                              MPI_COMM_WORLD, &request);
                    ++node_id;
                    ++active_nodes;
                } else {
                    MPI_Recv(buffer, buffer_size, MPI::INT, MPI_ANY_SOURCE,
                             tag::RESULT, MPI_COMM_WORLD, &status);
                    --active_nodes;
                    MPI_Isend(&i, 1, MPI::INT, status.MPI_SOURCE, tag::DATA,
                              MPI_COMM_WORLD, &request);
                    ++active_nodes;
                    if (__builtin_expect(buffer[0] == (height - 1), false)) {
                        std::copy_n((buffer + 1), (buffer_size - 1) / 2,
                                    (image + buffer[0] * width));
                    } else {
                        std::copy_n((buffer + 1), (buffer_size - 1),
                                    (image + buffer[0] * width));
                    }
                }
            }
            while (active_nodes > 0) {
                int info(-1);
                MPI_Recv(buffer, buffer_size, MPI::INT, MPI_ANY_SOURCE,
                         tag::RESULT, MPI_COMM_WORLD, &status);
                --active_nodes;
                MPI_Isend(&info, 1, MPI::INT, status.MPI_SOURCE, tag::TERMINATE,
                          MPI_COMM_WORLD, &request);
                if (__builtin_expect(buffer[0] == (height - 1), true)) {
                    std::copy_n((buffer + 1), (buffer_size - 1) / 2,
                                (image + buffer[0] * width));
                } else {
                    std::copy_n((buffer + 1), (buffer_size - 1),
                                (image + buffer[0] * width));
                }
            }
        }
        write_png(filename, iters, width, height, image);
        delete[](buffer);
        delete[](image);
    } else {
        int *buffer(new int[buffer_size]);
        // Worker
        int i(0);
        // Receive from manager
        MPI_Recv(&i, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        while (status.MPI_TAG != tag::TERMINATE) {
            buffer[0] = i;
            double y0(i * dy + lower);
#pragma omp parallel for schedule(dynamic, 100) collapse(2) 
{
            for (int tile_idx = 0; tile_idx < tile; ++tile_idx) {
                for (int j = 0; j < width; ++j) {
                    y0 = ((i + tile_idx) * dy + lower);
                    int offset(1 + tile_idx * width + j);
                    double x0(j * dx + left);
                    int repeats(0);
                    double x(0.0), y(0.0), length_squared(0.0);
                    while (repeats < iters && length_squared < 4.0) {
                        double temp(x * x - y * y + x0);
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    buffer[offset] = repeats;
                }
            }
}
            MPI_Send(buffer, buffer_size, MPI::INT, 0, tag::RESULT,
                     MPI_COMM_WORLD);
            MPI_Recv(&i, 1, MPI::INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        }
        delete[](buffer);
    }
    MPI_Finalize();
    return 0;
}

void write_png(const char *filename, int iters, int width, int height,
               const int *buffer) {
    FILE *fp = fopen(filename, "wb");
    png_structp png_ptr =
        png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);
    size_t row_size = 3 * width * sizeof(png_byte);
    png_bytep row = (png_bytep)malloc(row_size);
#pragma clang loop unroll(enable)
    for (int y = 0; y < height; ++y) {
        std::fill(row, row + row_size, 0);
#pragma clang loop vectorize(enable)
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