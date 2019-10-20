CXX:= clang++
exe:= test
obj:=test_ltbb.o
result := profile.csv profile.yml
all:
	clang++ -std=c++17 -march=native -O3 test_sort.cpp -ltbb -o test
.PHONY:clean
clean:
	rm -rf $(obj) $(exe) $(result)
