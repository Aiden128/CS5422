#ifndef SEQ_CHECK
#define SEQ_CHECK

#include <iostream>

void seq_check(bool print,int iters, int left, int right, int lower, int upper, int width, int height, int* check_target) {
    /* allocate memory for image */
    int* image = (int*)malloc(width * height * sizeof(int));

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
    for (int i = 0; i < width*height; ++i) {
        if(image[i] != check_target[i]) {
            ++diff_count;
            if(print == true) {
                std::cout << "Ref[" << i << "]: " << image[i];
                std::cout << " | Image[" << i << "]: " << check_target[i] << std::endl;
            }
        }
    }
    std::cout << "Diff count: " << diff_count << std::endl;
    free(image);
}

#endif