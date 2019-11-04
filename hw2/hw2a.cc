#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <chrono>
#include <complex>
#include <iostream>
#include <limits>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <cassert>

static void *producer(void *data);
void process_mandelbrot_set();
void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

struct schedule_data {
    int x0, y0, x1, y1;
    schedule_data(int x0, int y0, int x1, int y1) : 
    x0(x0), y0(y0), x1(x1), y1(y1) {}
};

double left, right;
double lower, upper;
int width, height;
int iters;
int num_thread;
int *image(nullptr);
int scheduled_idx(0);

const int tile_size = 200;
pthread_mutex_t *mutex;

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
    mutex = new (pthread_mutex_t);
    pthread_mutex_init(mutex, NULL);

    process_mandelbrot_set();
    write_png(filename, iters, width, height, image);

    delete[](image);
    delete[](mutex);

    return 0;
}

static void *producer(void *data) {
    while (1) {
        pthread_mutex_lock(mutex);
        if (scheduled_idx >= (width * height)) {
            pthread_mutex_unlock(mutex);
            break;
        }
        int start = scheduled_idx;
        scheduled_idx += tile_size;
        std::cout << "Thread schedule_idx: " << start <<std::endl;
        std::cout << "Updated idx: " << scheduled_idx << std::endl;
        pthread_mutex_unlock(mutex);

        schedule_data job_data(0, 0, 0, 0);
        job_data.x0 = ((start) / height);
        job_data.y0 = (start) % height;
        start += tile_size;
        job_data.x1 = ((start-1) / height);
        job_data.y1 = (start-1) % height;
        if(start >= width * height) {
            job_data.x1 = width - 1;
            job_data.y1 = height - 1;
        }
        std::cout << "Schedule x0,y0,x1,y1: "<< job_data.x0 << ", "<< job_data.y0 << ", "<< job_data.x1 << ", "<< job_data.y1 << std::endl;

        if(job_data.x0 == job_data.x1) {
            for (int j = job_data.y0; j <= job_data.y1; ++j) {
                double y0 = j * ((upper - lower) / height) + lower;
                for (int i = job_data.x0; i <= job_data.x1; ++i) {
                    double x0 = i * ((right - left) / width) + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4.0) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[j * width + i] = repeats;
                }
            }
        } else {
            for (int j = job_data.y0; j < height; ++j) {
                double y0 = j * ((upper - lower) / height) + lower;
                for(int i = job_data.x0; i <= job_data.x0; ++i) {
                    double x0 = i * ((right - left) / width) + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4.0) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[j * width + i] = repeats;
                }
            }
            for (int count = 0; count < (job_data.x1 - job_data.x0 - 1); ++count) {
                for (int j = 0; j <= height; ++j) {
                    double y0 = j * ((upper - lower) / height) + lower;
                    for (int i = job_data.x0 + 1; i < job_data.x1; ++i) {
                        double x0 = i * ((right - left) / width) + left;
                        int repeats = 0;
                        double x = 0;
                        double y = 0;
                        double length_squared = 0;
                        while (repeats < iters && length_squared < 4.0) {
                            double temp = x * x - y * y + x0;
                            y = 2 * x * y + y0;
                            x = temp;
                            length_squared = x * x + y * y;
                            ++repeats;
                        }
                        image[j * width + i] = repeats;
                    }
                }
            }
            for (int j = 0; j <= job_data.y1; ++j) {
                double y0 = j * ((upper - lower) / height) + lower;
                for(int i = job_data.x1; i <= job_data.x1; ++i) {
                    double x0 = i * ((right - left) / width) + left;
                    int repeats = 0;
                    double x = 0;
                    double y = 0;
                    double length_squared = 0;
                    while (repeats < iters && length_squared < 4.0) {
                        double temp = x * x - y * y + x0;
                        y = 2 * x * y + y0;
                        x = temp;
                        length_squared = x * x + y * y;
                        ++repeats;
                    }
                    image[j * width + i] = repeats;
                }
            }
        }
    }
    pthread_exit(NULL);
}

void process_mandelbrot_set() {
    pthread_t producer_threads[num_thread];

    for (int i = 0; i < num_thread; i++) {
        pthread_create(&producer_threads[i], NULL, producer, NULL);
        pthread_join(producer_threads[i], NULL);
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