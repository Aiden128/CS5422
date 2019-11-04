#ifndef PERF_HPP
#define PERF_HPP
// Use C++17 std
#include <chrono>
#include <fstream>
#include <string>
#include <map>
using namespace std;

using Map = std::map<std::string, chrono::time_point>;
using Record = std::map<std::string, double>;
class PERF{
public:
    explicit PERF();
    ~PERF();
    void start(const std::string &item_name);
    void end(const std::string &item_name);
    void dump_to_yaml(const std::string &file_name);
    void dump_to_csv(const std::string &file_name);
    void dump_to_stdout(const std::string &item_name);
    void clear();
private:
    map<std::string, chrono::time_point> start_dict;
    std::ofstream file;
};




#endif