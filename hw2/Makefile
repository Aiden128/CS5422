CC = clang
CXX = clang++ -m64
LDLIBS = -lpng -fopenmp -lpthread
CXXFLAGS = -Iobjs -lm -O3 -march=native -std=c++17

APP_NAME=hw2a
OBJDIR=objs

all: hw2a hw2b

hw2a:
	clang++ -Ofast -no-pie -flto hw2a.cc objs/mandelbrot_ispc.o -Iobjs -lpng -lpthread -lm -o hw2a

hw2b:
	mpicxx -cxx=clang++ -Ofast -fsanitize=address -no-pie hw2b.cc objs/mandelbrot_ispc.o -Iobjs -lpng -lpthread -fopenmp -lm -o hw2b

.PHONY: clean
clean:
	rm -f hw2a hw2b
