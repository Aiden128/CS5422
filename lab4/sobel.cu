#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "utils.h"

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

const char filter[Z][Y][X] = { { { -1, -4, -6, -4, -1 },
                                 { -2, -8, -12, -8, -2 },
                                 { 0, 0, 0, 0, 0 },
                                 { 2, 8, 12, 8, 2 },
                                 { 1, 4, 6, 4, 1 } },
                               { { -1, -2, 0, 2, 1 },
                                 { -4, -8, 0, 8, 4 },
                                 { -6, -12, 0, 12, 6 },
                                 { -4, -8, 0, 8, 4 },
                                 { -1, -2, 0, 2, 1 } } };

inline int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

void sobel(unsigned char *s, unsigned char *t, unsigned height, unsigned width,
           unsigned channels) {
    double val[Z][3];
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            /* Z axis of filter */
            for (int i = 0; i < Z; ++i) {

                val[i][2] = 0.;
                val[i][1] = 0.;
                val[i][0] = 0.;

                /* Y and X axis of filter */
                for (int v = -yBound; v <= yBound; ++v) {
                    for (int u = -xBound; u <= xBound; ++u) {
                        if (bound_check(x + u, 0, width) &&
                            bound_check(y + v, 0, height)) {
                            const unsigned char R =
                                s[channels * (width * (y + v) + (x + u)) + 2];
                            const unsigned char G =
                                s[channels * (width * (y + v) + (x + u)) + 1];
                            const unsigned char B =
                                s[channels * (width * (y + v) + (x + u)) + 0];
                            val[i][2] += R * filter[i][u + xBound][v + yBound];
                            val[i][1] += G * filter[i][u + xBound][v + yBound];
                            val[i][0] += B * filter[i][u + xBound][v + yBound];
                        }
                    }
                }
            }
            double totalR = 0.;
            double totalG = 0.;
            double totalB = 0.;
            for (int i = 0; i < Z; ++i) {
                totalR += val[i][2] * val[i][2];
                totalG += val[i][1] * val[i][1];
                totalB += val[i][0] * val[i][0];
            }
            totalR = sqrt(totalR) / SCALE;
            totalG = sqrt(totalG) / SCALE;
            totalB = sqrt(totalB) / SCALE;
            const unsigned char cR = (totalR > 255.) ? 255 : totalR;
            const unsigned char cG = (totalG > 255.) ? 255 : totalG;
            const unsigned char cB = (totalB > 255.) ? 255 : totalB;
            t[channels * (width * y + x) + 2] = cR;
            t[channels * (width * y + x) + 1] = cG;
            t[channels * (width * y + x) + 0] = cB;
        }
    }
}

int main(int argc, char **argv) {
    assert(argc == 3);
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst;
    /* read the image to src, and get height, width, channels */
    read_png(argv[1], &src, &height, &width, &channels);
    dst = (unsigned char *)malloc(height * width * channels *
                                  sizeof(unsigned char));
    /* computation */
    sobel(src, dst, height, width, channels);
    write_png(argv[2], dst, height, width, channels);
    return 0;
}
