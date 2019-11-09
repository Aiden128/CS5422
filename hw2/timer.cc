#include "timer.hpp"

Timer::Timer() = default;
Timer::~Timer() = default;

void Timer::start(const std::string &item_name) {
    auto _start(std::chrono::high_resolution_clock::now());
    start_dict.insert(std::pair(item_name, _start));
}

void Timer::end(const std::string &item_name) {
    auto _end(std::chrono::high_resolution_clock::now());
    auto _start(start_dict.find(item_name));
    if(_start != start_dict.end()) {
        auto elapsed_time(std::chrono::duration_cast<std::chrono::nanoseconds> (_end - _start->second).count());
        database.insert(std::pair(item_name, elapsed_time));
        start_dict.erase(item_name);
    } else {
        std::cerr << item_name << " not found in dict!" << std::endl;
        exit(-1);
    }
}

void Timer::print_stdout(const std::string &item_name) {
    auto _entry(database.find(item_name));
    if (_entry != database.end()) {
        std::cout << item_name << " elapsed time: " << _entry->second << " ns" << std::endl;
    } else {
        std::cerr << item_name << " not found!" << std::endl;
    }
}

void Timer::clear() {
    start_dict.clear();
    database.clear();
}