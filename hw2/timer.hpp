#ifndef TIMER_HPP
#define TIMER_HPP
// Use C++17 std
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <string>

using Entry =
    std::map<std::string, std::chrono::high_resolution_clock::time_point>;
using Record = std::map<std::string, double>;

class Timer {
  public:
    explicit Timer() = default;
    ~Timer() = default;
    inline void start(const std::string &item_name) {
        auto _start(std::chrono::high_resolution_clock::now());
        start_dict.insert(std::pair(item_name, _start));
    };
    inline void end(const std::string &item_name) {
        auto _end(std::chrono::high_resolution_clock::now());
        auto _start(start_dict.find(item_name));
        if (_start != start_dict.end()) {
            auto elapsed_time(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    _end - _start->second)
                    .count());
            database.insert(std::pair(item_name, elapsed_time));
            start_dict.erase(item_name);
        } else {
            std::cerr << item_name << " not found in dict!" << std::endl;
            exit(-1);
        }
    };
    inline void print_stdout(const std::string &item_name) {
        auto _entry(database.find(item_name));
        if (_entry != database.end()) {
            std::cout << item_name << " elapsed time: " << _entry->second
                      << " ns" << std::endl;
        } else {
            std::cerr << item_name << " not found!" << std::endl;
        }
    };
    inline void clear() {
        start_dict.clear();
        database.clear();
    };
    // TODO
    // void dump_to_yaml(const std::string &file_name);
    inline void dump_csv(const std::string &file_name) {
        file.open(file_name, std::ofstream::out | std::ofstream::app);
        for(auto i : database) {
            file << i.first << "," << i.second << "," << std::endl;
        }
        file.close();
    };
  private:
    Entry start_dict;
    Record database;
    std::ofstream file;
};
#endif