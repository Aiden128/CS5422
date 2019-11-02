#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#define DEBUG
#ifdef DEBUG
#include <cassert>
#endif
#include <iostream>
#include <memory>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <vector>

using namespace std;

void *Op(void *threadD);
void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

struct thread_data {
    int threadID;
    int num_thread;
    int *image;
    double left, right;
    double lower, upper;
    int width, height;
    int iter;
};

int main(int argc, char **argv) {

#ifdef DEBUG
    assert(argc == 9);
#endif

    // Get number of CPUs available
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);

    const int num_thread(CPU_COUNT(&cpu_set));
    const char *filename(argv[1]);
    const long iters(strtol(argv[2], 0, 10));
    const double left(strtod(argv[3], 0));
    const double right(strtod(argv[4], 0));
    const double lower(strtod(argv[5], 0));
    const double upper(strtod(argv[6], 0));
    const long width(strtol(argv[7], 0, 10));
    const long height(strtol(argv[8], 0, 10));
    int *image(new int[width * height]{0});
    //shared_ptr<int> image(new int[width*height]{0});
    pthread_t *threads(new pthread_t[num_thread]);
    thread_data *threadD(new thread_data[num_thread]);

    for (int i = 0; i < num_thread; ++i) {
        threadD[i].threadID = i;
        threadD[i].num_thread = num_thread;
        threadD[i].image = image;
        threadD[i].left = left;
        threadD[i].right = right;
        threadD[i].lower = lower;
        threadD[i].upper = upper;
        threadD[i].width = width;
        threadD[i].height = height;
        threadD[i].iter = iters;
        // Create threads
        pthread_create(&threads[i], NULL, Op, (void *)&threadD[i]);
    }
    for (int i = 0; i < num_thread; ++i) {
        pthread_join(threads[i], NULL);
    }
    write_png(filename, iters, width, height, image);

    delete[] image;
    delete[] threads;
    delete[] threadD;
    return 0;
}

void *Op(void *threadD) {
    thread_data *args = reinterpret_cast<thread_data *>(threadD);
    int totalRows = args->height / args->num_thread;
    int res = args->height % args->num_thread;
    int startRow = totalRows * args->threadID;
    if(args->threadID == 0) {
        startRow = 0;
        totalRows += res;
    }
    else {
        startRow += res;
    }
    int endRow = startRow + totalRows;

    for (int j = startRow; j < endRow; ++j) {
        double y0 = j * ((args->upper - args->lower) / args->height) + args->lower;
        for (int i = 0; i < args->width; ++i) {
            double x0 = i * ((args->right - args->left) / args->width) + args->left;
            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < args->iter && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            args->image[j * args->width + i] = repeats;
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
        fill(row, row + row_size, 0);
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
