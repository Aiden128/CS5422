#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <pthread.h>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <algorithm>
#include "util.h"

pthread_t *threads;
pthread_mutex_t mutex, ftex;
int grain = 100;
int thread_count;
int queue_size = 20;
int head = 0;
int tail = 0;
int count = 0; 
int done = 0;
int tasks = 25;


int* image;
int iters, width, height;
double left, right, lower, upper;
int xslice, yslice;
double dx, dy; 
typedef struct {
    int x;
    int y;
} inarg;
inarg *argvs;
void* worker(void* arg);
int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    thread_count = CPU_COUNT(&cpu_set);
    /* argument parsing */
    assert(argc == 9);
    const char* filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);
    // slice = height / tasks;
    dx = (double)((right - left) / width);
    dy = (double)((upper - lower) / height);
    
    image = (int*)malloc(width * height * sizeof(int));
    assert(image);

    threads = (pthread_t*)malloc(sizeof(pthread_t)*thread_count);
    queue = (threadpool_task_t*)malloc(sizeof(threadpool_task_t)*queue_size);
    argvs = (inarg*)malloc(sizeof(inarg) * queue_size);
    
    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&mutex, NULL);
    pthread_mutex_init(&ftex, NULL);
    for (int i = 0; i < thread_count; i++)
        pthread_create(&threads[i], NULL, worker, NULL);
    
        
    for (int i = 0; i < thread_count; i++)
        pthread_join(threads[i], NULL);
    printf("count: %d\n", count);
    pthread_mutex_destroy(&mutex);
    pthread_mutex_destroy(&ftex);
    pthread_cond_destroy(&cond);
    write_png(filename, iters, width, height, image);
    free(image);
    free(threads);
    free(queue);
    free(argvs);
    return EXIT_SUCCESS;
}

void* worker(void* arg) { 
    int lf; 
    int flag = 0; // last flag
    for (;;) {
        pthread_mutex_lock(&mutex);
        if(count == width * height) { pthread_mutex_unlock(&mutex); break; }
        lf = count;
        if(count + grain < width * height) 
            count += grain; 
        else {
            count = width * height; 
            flag = 1;
        }
        pthread_mutex_unlock(&mutex);
        int end = flag ? width * height : lf + grain;
        for(int c = lf; c < end; c++) {
            int j = c / width;
            int i = c % width;
            double y0 = j * dy + lower;
            double x0 = i * dx  + left;
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
            image[c] = repeats;
        }
        if(flag) { break; }
        
    }

    pthread_exit(NULL);
}
