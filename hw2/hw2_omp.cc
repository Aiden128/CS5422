#include "timer.hpp"
#include <cassert>
#include <chrono>
#include <complex>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <png.h>
using namespace std;

void write_png(const char *filename, int iters, int width, int height,
               const int *buffer);

template <typename T, typename U>
inline std::complex<T> operator*(const std::complex<T> &lhs, const U &rhs) {
    return lhs * T(rhs);
}

template <typename T, typename U>
inline std::complex<T> operator*(const U &lhs, const std::complex<T> &rhs) {
    return T(lhs) * rhs;
}

int main(int argc, char **argv) {

    assert(argc == 9);

    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);
    int image_size(width * height);
    /* allocate memory for image */
    int *image1(new int[image_size]);
    double dx = (right - left) / width;
    double dy = (upper - lower) / height;

    Timer timer;

    timer.start("omp_normal");
#pragma omp parallel for schedule(dynamic, 100)
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
        image1[pixel_idx] = repeats;
    }

    timer.end("omp_normal");
    timer.start("omp_simd");

#pragma omp parallel for simd schedule(dynamic, 100)
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
        image1[pixel_idx] = repeats;
    }
    timer.end("omp_simd");

    timer.print_stdout("omp_normal");
    timer.print_stdout("omp_simd");

    write_png(filename, iters, width, height, image1);
    delete[](image1);
    return 0;
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
        // memset(row, 0, row_size);
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