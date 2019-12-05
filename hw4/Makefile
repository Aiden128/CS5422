NVFLAGS  := -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 
LDFLAGS  := -lm
EXES     := seq apsp

alls: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

apsp: apsp.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?
