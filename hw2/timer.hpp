#ifndef TIMER_HPP
#define TIMER_HPP
// Use C++17 std
#include <chrono>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <set>

using Entry = std::map<std::string, std::chrono::high_resolution_clock::time_point>;
using Record = std::map<std::string, double>;

class Timer{
public:
    explicit Timer();
    ~Timer();
    void start(const std::string &item_name);
    void end(const std::string &item_name);
    void print_stdout(const  std::string &item_name);
    void clear();
    // TODO
    // void dump_to_yaml(const std::string &file_name);
    // void dump_to_csv(const std::string &file_name);
private:
    Entry start_dict;
    Record database;
    std::ofstream file;
};
#endif