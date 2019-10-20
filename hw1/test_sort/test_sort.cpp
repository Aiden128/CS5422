#include "tbb/task_scheduler_init.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <execution>
#include <fstream>

using namespace std;

void printDuration(std::chrono::steady_clock::time_point start, std::chrono::steady_clock::time_point end, const char *message) {
        auto diff = end - start;
        std::cout << message << ' ' << std::chrono::duration <double, std::milli> (diff).count() << " ms\n";
}

template<typename T>
void test(const T &policy, const std::vector<float> &data, const int repeat, const char *message) {
    double elap_time(0.0);
    double avg_elap_time(0.0);
    std::ofstream file;
    std::ofstream csv_file;
    for(int i = 0; i < repeat; ++i) {
        std::vector<float> curr_data(data);
        const auto start = std::chrono::high_resolution_clock::now();
        std::sort(policy, curr_data.begin(), curr_data.end());
        const auto end = std::chrono::high_resolution_clock::now();
        elap_time += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    }
    avg_elap_time = elap_time / repeat;
    auto sample_count = data.size();

    file.open("profile.yml", std::ofstream::out | std::ofstream::app);
    csv_file.open("profile.csv", std::ofstream::out | std::ofstream::app);
    file << sample_count << ":" << std::endl;
    if constexpr(std::is_same_v<T, std::execution::sequenced_policy>) {
        file << "    " << "policy: sequential" << endl;
        csv_file << avg_elap_time << ", ";
    } else {
        file << "    " << "policy: parallel" << endl;
        csv_file << avg_elap_time << "," << endl;
    }
    file << "    " << "average_time: " << avg_elap_time << endl;
    file.close(); 
    csv_file.close();
}

int main() {
    // Test samples and repeat factor
    // constexpr size_t sample_vec[37] = {
    //     1, 5, 10, 50, 100, 500, 1000, 5000, 
    // 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
    // 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
    // 1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 
    // 5000000, 5500000, 6000000};
    constexpr size_t sample_vec[8] = {
        6500000, 7000000, 7500000, 8000000,
        8500000, 9000000, 9500000, 10000000
    };
    constexpr int repeat{100};

    // Fill a vector with samples numbers
    std::random_device rd;
    std::mt19937_64 mre(rd());
    std::uniform_real_distribution<double> urd(0.0, 1.0);
    //tbb::task_scheduler_init init(2);

    for (int i = 0; i < 8; ++i) {
        std::vector<float> data(sample_vec[i]);
        for(auto &e : data) {
            e = urd(mre);
        }
        std::cout << "Samples: " << sample_vec[i] << std::endl;
        // Sort data using different execution policies
        std::cout << "std::execution::seq\n";
        test(std::execution::seq, data, repeat, "Elapsed time");

        std::cout << "std::execution::par\n";
        test(std::execution::par, data, repeat, "Elapsed time");
    }

}