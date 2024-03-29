#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "objs/mandelbrot_ispc.h"
#include "util.h"
#include <iostream>
#include <limits>
#include <png.h>
#include <pthread.h>
#include <sched.h>

static void *producer(void *data);
void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

SetOnce<double> left, right;
SetOnce<double> lower, upper;
SetOnce<double> dx, dy;
SetOnce<int> width, height;
SetOnce<int> image_size, iters, num_thread;
int *image(nullptr);
int scheduled_idx(0);

const int tile_size(50);
pthread_spinlock_t lock;

int main(int argc, char **argv) {

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
    image_size = width * height;
    image = new int[image_size];
    pthread_spin_init(&lock, 0);
    pthread_t threads[num_thread.get()];
    cpu_set_t thread_cpu[num_thread.get()];
    dx = static_cast<double>((right - left) / width);
    dy = static_cast<double>((upper - lower) / height);

    for (int i = 0; i < num_thread; ++i) {
        pthread_create(&threads[i], NULL, producer, NULL);
        CPU_ZERO(&thread_cpu[i]);
        CPU_SET(i, &thread_cpu[i]);
        pthread_setaffinity_np(threads[i], sizeof(cpu_set_t), &thread_cpu[i]);
    }
    for (int i = 0; i < num_thread; ++i) {
        pthread_join(threads[i], NULL);
    }
    write_png(filename, iters, width, height, image);

    delete[](image);
    return 0;
}

static void *producer(void *data) {
    int start_idx(0), end_idx(0);
    bool end_flag(false);
    while (1) {
        pthread_spin_lock(&lock);
        if (__builtin_expect(scheduled_idx == (image_size), false)) {
            pthread_spin_unlock(&lock);
            break;
        }
        start_idx = scheduled_idx;
        if (__builtin_expect((start_idx + tile_size < image_size), true)) {
            scheduled_idx += tile_size;
        } else {
            scheduled_idx = image_size;
            end_flag = true;
        }
        pthread_spin_unlock(&lock);
        if (__builtin_expect(end_flag, false)) {
            end_idx = image_size;
        } else {
            end_idx = start_idx + tile_size;
        }
        ispc::mandelbrot_ispc(left, lower, dx, dy, width, iters, start_idx,
                              end_idx, image);
        if (__builtin_expect((end_flag), false)) {
            break;
        }
    }
    pthread_exit(NULL);
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