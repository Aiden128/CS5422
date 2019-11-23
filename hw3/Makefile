CC = clang
CXX = clang++
CXXFLAGS = -Ofast -march=native -ltbb -fopenmp
CFLAGS = -Ofast -lm -pthread
TARGETS = hw3

#.PHONY: all
#all: $(TARGETS)

hw3:
	clang++ -Ofast -no-pie -std=c++17 -Wall hw3_tbb.cc objs/floyd_warshall.o  -Iobjs -ltbb -fopenmp -lm -o hw3

.PHONY: clean
clean:
	rm -f $(TARGETS)
