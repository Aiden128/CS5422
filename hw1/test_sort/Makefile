CXX:= clang++
exe:= test
obj:=test_ltbb.o

all:
	clang++ -std=c++17 -O3 test_sort.cpp -ltbb -o test
.PHONY:clean
clean:
	rm -rf $(obj) $(exe)
