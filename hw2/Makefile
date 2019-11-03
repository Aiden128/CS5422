CC = clang
CXX = clang++
LDLIBS = -lpng
CXXFLAGS = -lm -O3
hw2a: CXXFLAGS += -pthread -O3 -march=native
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CXXFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
