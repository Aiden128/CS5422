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

static void *producer(void *data);
void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

double left, right;
double lower, upper;
double dx, dy;
int width, height;
int image_size;
int iters;
int num_thread;
int *image(nullptr);
int scheduled_idx(0);

const int tile_size = 100;
pthread_mutex_t mutex;

int main(int argc, char **argv) {
    // allow complex literal
    using namespace std::literals;
    // Get number of CPUs available
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    num_thread = (CPU_COUNT(&cpu_set));
    const char *filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    image = new int[width * height];
    pthread_mutex_init(&mutex, NULL);
    pthread_t producer_threads[num_thread];

    image_size = width * height;
    dx = static_cast<double> ((right - left) / width);
    dy = static_cast<double> ((upper - lower) / height);

    for (int i = 0; i < num_thread; ++i) {
        pthread_create(&producer_threads[i], NULL, producer, NULL);
    }
#ifdef PERF
    auto comp_start = std::chrono::high_resolution_clock::now();
#endif
    for (int i = 0; i < num_thread; ++i) {
        pthread_join(producer_threads[i], NULL);
    }
#ifdef PERF
    auto comp_end = std::chrono::high_resolution_clock::now();
    auto comp_time = std::chrono::duration_cast<std::chrono::nanoseconds>(comp_end - comp_start).count();
    std::cout << "Computation time: " << comp_time << " ns" << std::endl;
#endif
    write_png(filename, iters, width, height, image);

    delete[](image);

    return 0;
}

static void *producer(void *data) {
    int start_idx(0), end_idx(0);
    bool end_flag(false);
    while (1) {
        pthread_mutex_lock(&mutex);
        if (scheduled_idx == (image_size)) {
            pthread_mutex_unlock(&mutex);
            break;
        }
        start_idx = scheduled_idx;
        if(start_idx + tile_size < image_size) {
            scheduled_idx += tile_size;
        } else {
            scheduled_idx = image_size;
            end_flag = true;
        }
        pthread_mutex_unlock(&mutex);
        if(end_flag) {
            end_idx = image_size;
        } else {
            end_idx = start_idx + tile_size;
        }
        for(auto pixel : boost::irange(start_idx, end_idx)) {
            int j(pixel / width), i(pixel % width);
            double y0(j * dy + lower), x0(i * dx + left);
            int repeats(0);
            double x(0.0), y(0.0), length_squared(0.0);
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[pixel] = repeats;
        }
        if(end_flag) {
            break;
        }
    }
    pthread_exit(NULL);
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