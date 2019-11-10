#include <boost/range/irange.hpp>
#include <chrono>
#include <complex>
#include <iostream>
#include <limits>
#include <png.h>
#include <tbb/tbb.h>
#include <tiffio.h>

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

template <typename Iterate, typename IterationMap, typename T>
int boundedorbit(
    Iterate f, std::complex<T> seed, int bound, int bailout,
    IterationMap itmap = [](int n, std::complex<T> z, int bailout) {
        return n;
    }) {
    auto z = f(seed);
    for (auto k : boost::irange(1, bailout)) {
        if (abs(z) > bound) {
            return k;
        }
        z = f(z);
    }
    return bailout;
}

template <typename T>
float normalized_iterations(int n, std::complex<T> zn, int bailout) {
    return n + (log(log(bailout)) - log(log(abs(zn)))) / log(2);
}

int main(int argc, char **argv) {
    // allow complex literal
    using namespace std::literals;
    // initialize TBB
    tbb::task_scheduler_init init;
    const int GRAIN = 200;

    const char *filename = argv[1];
    int iters = strtol(argv[2], 0, 10);
    double left = strtod(argv[3], 0);
    double right = strtod(argv[4], 0);
    double lower = strtod(argv[5], 0);
    double upper = strtod(argv[6], 0);
    int width = strtol(argv[7], 0, 10);
    int height = strtol(argv[8], 0, 10);

    // allocate storage
    int *iteration_counts = new int[width * height];
    // auto start = std::chrono::high_resolution_clock::now();

    double dx = (right - left) / width;
    double dy = (upper - lower) / height;
    tbb::parallel_for(
        tbb::blocked_range2d<int>(0, height, 50, 0, width, 2), [&](auto r) {
            for (auto j = r.rows().begin(); j != r.rows().end(); j++) {
                for (auto k = r.cols().begin(); k != r.cols().end(); k++) {
                    std::complex<double> c((left + k * dx), (lower + j * dy));
                    iteration_counts[width * j + k] =
                        boundedorbit([&c](auto z) { return z * z + c; }, 0.0i,
                                     2, iters, &normalized_iterations<double>);
                }
            }
        });

    // tbb::parallel_for(tbb::blocked_range<int>(0, height, GRAIN), [&](auto r)
    // {
    //     for (auto j = r.begin(); j != r.end(); j++) {
    //         for (auto k = 0; k < width; k++) {
    //             std::complex<double> c ((left + k*dx), (lower + j*dy));
    //             iteration_counts[width*j + k] = boundedorbit([&c](auto z) {
    //             return z*z + c; }, 0.0i, 2, iters,
    //             &normalized_iterations<double>);
    //         }
    //     }
    // });

    // auto end = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> elapsed_seconds = end - start;

    // std::cout << "computation took " << elapsed_seconds.count() << "s" <<
    // std::endl;

    write_png(filename, iters, width, height, iteration_counts);

    delete[](iteration_counts);
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