#include <cassert>
#include <chrono>
#include <complex>
#include <cstring>
#include <iostream>
#include <png.h>
using namespace std;

static inline int mandel(const double &c_re, const double &c_im,
                         const int &count);
inline int mandel(const complex<double> &c, const int &count);
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

    /* allocate memory for image */
    int *image1 = new int[height * width];

    double dx = (right - left) / width;
    double dy = (upper - lower) / height;

    // for (int j = 0; j < height; ++j) {
    //     for (int i = 0; i < width; ++i) {
    //         double x(left + i * dx);
    //         double y(lower + j * dy);
    //         int idx(j * width + i);
    //         image1[idx] = mandel(x, y, iters);
    //     }
    // }
    for (int j = 0; j < height; ++j) {
        for (int i = 0; i < width; ++i) {
            complex<double> c(left + i * dx, lower + j * dy);
            int idx(j * width + i);
            image1[idx] = mandel(c, iters);
        }
    }

    /* allocate memory for image */
    int *image = (int *)malloc(width * height * sizeof(int));
    assert(image);

    /* mandelbrot set */
    for (int j = 0; j < height; ++j) {
        double y0 = j * ((upper - lower) / height) + lower;
        for (int i = 0; i < width; ++i) {
            double x0 = i * ((right - left) / width) + left;

            int repeats = 0;
            double x = 0;
            double y = 0;
            double length_squared = 0;
            while (repeats < iters && length_squared < 4) {
                double temp = x * x - y * y + x0;
                y = 2 * x * y + y0;
                x = temp;
                length_squared = x * x + y * y;
                ++repeats;
            }
            image[j * width + i] = repeats;
        }
    }

    int diff_count(0);
    double rate(0.0);
    for (int i = 0; i < width * height; ++i) {
        if (image[i] != image1[i]) {
            ++diff_count;
            cout << "ref[" << i << "]: " << image[i] << " | ";
            cout << "image[" << i << "]: " << image1[i] << endl;
        }
    }

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < height; j++) {
            std::cout << image1[i * width + j] << " ";
        }
        std::cout << std::endl;
    }
    rate = static_cast<double>(diff_count / (width * height));
    cout << "Err rate: " << diff_count << endl;

    write_png(filename, iters, width, height, image1);
    delete[](image);
    delete[](image1);
    return 0;
}

static inline int mandel(const double &c_re, const double &c_im,
                         const int &count) {
    double z_re(0.0), z_im(0.0);
    double new_re(0.0), new_im(0.0);
    int i(0);

    for (i = 0; i < count && (z_re * z_re + z_im * z_im < 4.0); ++i) {
        new_re = z_re * z_re - z_im * z_im;
        new_im = 2.0 * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

inline int mandel(const complex<double> &c, const int &count) {
    complex<double> z(0.0, 0.0);
    int i(0);
    while (i < count && abs(z) < 2) {
        z = z * z + c;
        ++i;
    }
    return i;
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
        memset(row, 0, row_size);
        // fill(row, row+row_size, 0);
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