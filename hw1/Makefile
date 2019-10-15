CXX := mpicxx
CXXFLAGS := -cxx=clang++ -O3 -Wall -std=c++17 -march=native
TARGET := hw1
SRCS := hw1.cc oe.cc
OBJS := hw1.o oe.o 

hw1: $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)
%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $^ -o $@
.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS)
