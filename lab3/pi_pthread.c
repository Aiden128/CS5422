#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
double distance(double x, double y);
void *Op (void *threadD);

typedef struct thread_data {
    int *in_circle;                    // number of points in the unit circle
    pthread_mutex_t *in_circle_lock;   // lock control
    int threadID;                      // thread ID
    long long num_tosses;              // load on each thread
}thread_data;

int main (int argc, char *argv[]){

    double pi;              // estimatd Pi
    int i;                  // loop index
    int points;             // user prompt, the number of generated points
    int numthreads;         // number of threads will be created
    int in_circle = 0;      // count the number of points inside the unit circle
    int iteration;          // number of iterations will run in each thread
    int remain;             // the rest iterations not distribute
    pthread_t* threads;     // thread variable
    thread_data* threadD;   // thread data pointer
    pthread_mutex_t in_circle_lock;     //mutex lock

    points  = atoi(argv[2]);            // read user input
    numthreads = atoi(argv[1]);         // read user input
    iteration = points / numthreads;    // distribute iterations
    remain = points % numthreads;
    threads = (pthread_t*)malloc(sizeof(pthread_t) * numthreads);       // create object
    threadD = (thread_data*)malloc(sizeof(thread_data) * numthreads);   // create object
    pthread_mutex_init(&in_circle_lock, NULL);          // mutex lock initialization
    for(i=0;i<numthreads;i++) {
        threadD[i].in_circle= &in_circle;               // set thread infomation
        threadD[i].in_circle_lock = &in_circle_lock;
        threadD[i].threadID = i;
        if(i == 0) threadD[i].num_tosses = iteration + remain;         //set load
        else threadD[i].num_tosses = iteration;
        pthread_create(&threads[i], NULL, Op, (void*)&threadD[i]);
    }
    for(i=0; i<numthreads; i++) {
        pthread_join(threads[i], NULL);
    }
    pi = 4.0 * ((double)in_circle/(double)(points));    //calculate Pi
    printf("Estimate of Pi is %f\n",pi);                //print result

    free(threads);
    free(threadD);

    return 0;

}

void *Op(void *threadD) {
    int i;
    double x, y;
    double dis;
    int count = 0;

    thread_data* tData = (thread_data*)threadD;
    srandom(time(NULL));
    for(i = 0; i < tData->num_tosses; i++) {
        x = (double)rand()/RAND_MAX;// generate location info on x axis
        y = (double)rand()/RAND_MAX;// generate location info on y axis
        dis = distance(x,y);        // find distance, use (0,0) as reference point
        if(dis < 1.0) count++;      // record points inside the unit circle
    }

    pthread_mutex_lock(tData->in_circle_lock);     // mutex lock on
    *(tData->in_circle) += count;                  // record
    pthread_mutex_unlock(tData->in_circle_lock);   // mutex lock off

    pthread_exit(NULL);
}

double distance(double x, double y) {
    double result;

    result = sqrt((x*x) + (y*y));   // cauculate by junior math
    return result;
}