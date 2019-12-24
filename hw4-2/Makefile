NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
CXXFLAGS := -fopenmp -w
LDFLAGS  := -lm
EXES     := hw4-2

alls: $(EXES)

clean:
	rm -f $(EXES)

hw4-2: test.cu
	nvcc $(NVFLAGS) -Xcompiler="$(CXXFLAGS)" $(LDFLAGS) -o $@ $?
