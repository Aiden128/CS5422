CC           = gcc
CFLAGS       = -O3 -march=native -std=c11 -Wall
LDFLAGS      = -L/home/pp19/share/lib -Wl,-rpath=/home/pp19/share/lib -llab4 -lm
NVCC         = nvcc 
NVCCFLAGS    = -O2 -std=c++11 -Xptxas=-v -arch=sm_61
NVCCLDFLAGS  = -L/home/pp19/share/lib -Xlinker=-rpath,/home/pp19/share/lib -llab4

TARGETS = sobel sobel.cuda

.PHONY: all
all: $(TARGETS)

soble: sobel.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

sobel.cuda: sobel.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCLDFLAGS) -o $@ $<

.PHONY: clean
clean:
	rm -f $(TARGETS)
