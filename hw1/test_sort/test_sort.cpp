#include "tbb/task_scheduler_init.h"
#include <boost/sort/spreadsort/float_sort.hpp>
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <chrono>
#include <execution>
#include <fstream>

using namespace std;


void test_boost(const std::vector<float> &data, const int repeat) {
    double elap_time(0.0);
    double avg_elap_time(0.0);
    std::ofstream file;
    std::ofstream csv_file;
    for(int i = 0; i < repeat; ++i) {
        std::vector<float> curr_data(data);
        const auto start = std::chrono::high_resolution_clock::now();
        boost::sort::spreadsort::float_sort(curr_data.begin(), curr_data.end());
        const auto end = std::chrono::high_resolution_clock::now();
        elap_time += chrono::duration_cast<chrono::nanoseconds>(end - start).count();
    }
    avg_elap_time = elap_time / repeat;
    auto sample_count = data.size();

    file.open("profile.yml", std::ofstream::out | std::ofstream::app);
    csv_file.open("profile.csv", std::ofstream::out | std::ofstream::app);
    file << sample_count << ":" << std::endl;
    file << "    " << "policy: boost" << endl;
    csv_file << avg_elap_time << "," << endl;
    file << "    " << "average_time: " << avg_elap_time << endl;
    file.close(); 
    csv_file.close();

};
template<typename T>
void test(const T &policy, const std::vector<float> &data, const int repeat) {
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
        csv_file << avg_elap_time << ",";
    }
    file << "    " << "average_time: " << avg_elap_time << endl;
    file.close(); 
    csv_file.close();
}

int main() {
    // Test samples and repeat factor
    constexpr size_t sample_vec[45] = {
    1, 5, 10, 50, 100, 500, 1000, 5000, 
    10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000,
    100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
    1000000, 1500000, 2000000, 2500000, 3000000, 3500000, 4000000, 4500000, 
    5000000, 5500000, 6000000, 6500000, 7000000, 7500000, 8000000,
    8500000, 9000000, 9500000, 10000000};

    constexpr int repeat{100};

    // Fill a vector with samples numbers
    std::random_device rd;
    std::mt19937_64 mre(rd());
    std::uniform_real_distribution<double> urd(0.0, 1.0);
    //tbb::task_scheduler_init init(2);

    for (int i = 0; i < 45; ++i) {
        std::vector<float> data(sample_vec[i]);
        for(auto &e : data) {
            e = urd(mre);
        }
        std::cout << "Samples: " << sample_vec[i] << std::endl;

        std::cout << "std::execution::seq\n";
        test(std::execution::seq, data, repeat);

        std::cout << "std::execution::par\n";
        test(std::execution::par, data, repeat);

        std::cout << "boost::float_sort\n";
        test_boost(data, repeat);
    }

}