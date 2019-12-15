NVFLAGS  := -std=c++11 -use_fast_math -Xcompiler -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
EXES     := seq hw4-1 baseline

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw4-1: hw4-1.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

baseline: apsp.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?