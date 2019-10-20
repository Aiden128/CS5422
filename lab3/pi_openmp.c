#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <time.h>
double distance(double x, double y);

int main (int argc, char *argv[]){

    double x,y;             // location infomation
    double pi;              // estimatd Pi
    double dis;             // distance between random location and (0,0)
    int i;                  // loop index
    int points;             // user prompt, the number of generated points
    int numthreads;         // number of threads will be created
    int count=0;            // count the number of points inside the unit circle

    points  = atoi(argv[1]);            // read user input
    numthreads = atoi(argv[2]);         // read user input
    #pragma omp parallel private(x,y,dis,i)shared(count) num_threads(numthreads)
    {
        srandom(time(NULL));            // random seed based on time
        for(i=0;i<points;i++){
            x = (double)rand()/RAND_MAX;// generate location info on x axis
            y = (double)rand()/RAND_MAX;// generate location info on y axis
            dis = distance(x,y);        // find distance, use (0,0) as reference point
            #pragma omp critical
            {
                if(dis < 1.0) count++;      // record points inside the unit circle
            }
        }
    }
    pi = 4.0*((double)count/(double)(points*numthreads));   //calculate Pi
    printf("Estimate of Pi is %f\n",pi); //print result

    return 0;

}

double distance(double x, double y){
    double result;

    result = sqrt((x*x) + (y*y));   // cauculate by junior math
    return result;
}