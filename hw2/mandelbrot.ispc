static inline int mandel(double c_re, double c_im, int count) {
    double z_re = 0, z_im = 0;
    int i;
    for (i = 0; i < count; ++i) {

        if (z_re * z_re + z_im * z_im > 4.f)
           break;

        double new_re = z_re * z_re - z_im * z_im;
        double new_im = 2.f * z_re * z_im;
        z_re = c_re + new_re;
        z_im = c_im + new_im;
    }

    return i;
}

export void mandelbrot_ispc(uniform double x0, uniform double y0, 
                            uniform double dx, uniform double dy,
                            uniform int width, uniform int maxIterations,
                            uniform int start, uniform int end,
                            uniform int output[])
{
    foreach (i = start ... end) {

        double x = (i % width) * dx + x0;
        double y = (i / width) * dy + y0;

        output[i] = mandel(x, y, maxIterations);
    }
}

export void mandelbrot_omp_ispc(uniform double x0, uniform double y0, 
                            uniform double dx, uniform double dy,
                            uniform int width, uniform int maxIterations,
                            uniform int start, uniform int end,
                            uniform int pixel_idx,
                            uniform int output[])
{
    foreach (i = start ... end) {

        double x = (pixel_idx % width) * dx + x0;
        double y = (pixel_idx / width) * dy + y0;

        output[i] = mandel(x, y, maxIterations);
    }
}