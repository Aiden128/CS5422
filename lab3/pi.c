#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv) {
  double pi = 0;
  long long numParts = atoll(argv[1]);

  for (long long i = 0; i < numParts; i++) {
    pi += sqrt(1 - ((double)i / numParts) * ((double)i / numParts));
  }

  printf("%.12lf\n", pi * 4 / numParts);

  return 0;
}
