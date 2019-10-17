// CS5540 Homework 1: Odd-Even Sort
// Author: Jerry ZJ
// Date: 2019/09/27
#include "oe.h"
#ifdef DEBUG
#include <cassert>
#endif
using namespace std;

int main(int argc, char **argv) {

#ifdef DEBUG
    assert(argc == 4); // Make sure arguments are ready
#endif
#ifdef PERF
    auto main_start = chrono::high_resolution_clock::now();
#endif
    int rank(0), task_num(0);
    // Start processing
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &task_num);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    OE_sort oe(rank, task_num, stoi(argv[1]), argv[2], argv[3]);

    if (oe.schedule == parallel) {
        oe.parallel_read_file();
        oe.parallel_sort();
        oe.parallel_write_file();
    } else if (oe.schedule == single) {
        oe.single_read_file();
        oe.single_sort();
        oe.single_write_file();
    }

#ifdef PERF
    double local_results[7], global_results[7];
    local_results[0] = oe.mem_time;
    local_results[1] = oe.read_time;
    local_results[2] = oe.write_time;
    local_results[3] = oe.MPI_transmission_time;
    local_results[4] = oe.MPI_sync_time;
    local_results[5] = oe.merge_time;
    local_results[6] = oe.stl_sort_time;
    MPI_Reduce(&local_results, &global_results, 7, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    // cout << "Rank: " << rank << " read: " << oe.read_time << " ns" << endl;
    // cout << "Rank: " << rank << " write: " << oe.write_time << " ns" << endl;
    if (rank == 0) {
        if (oe.schedule == parallel) {
            for (int i = 0; i < 7; ++i) {
                global_results[i] /= task_num;
            }
        }
        ofstream file;
        ofstream csv_file;
        string test_filename(argv[2]);
        size_t pos = test_filename.find("../");
        if (pos != std::string::npos) {
            // If found then erase it from string
            test_filename.erase(pos, 3);
        }
        file.open("profile.yml", std::ofstream::out | std::ofstream::app);
        file << test_filename << ":" << endl; 
        file << "    Mem allocate: " << global_results[0] << endl;
        file << "    MPI Read: " << global_results[1]  << endl;
        file << "    MPI Write: " << global_results[2] << endl;
        file << "    MPI Trans: " << global_results[3] << endl;
        file << "    MPI Sync: " << global_results[4] << endl;
        file << "    Merge: " << global_results[5] << endl;
        file << "    STL sort: " << global_results[6] << endl;
        auto avg_total = std::accumulate(global_results, global_results + 7, 0.0);
        file << "    Total: " << avg_total << endl;
        file.close();
        csv_file.open("profile.csv", std::ofstream::out | std::ofstream::app);
        csv_file << global_results[0] << ",";
        csv_file << global_results[1] << ",";
        csv_file << global_results[2] << ",";
        csv_file << global_results[3] << ",";
        csv_file << global_results[4] << ",";
        csv_file << global_results[5] << ",";
        csv_file << global_results[6] << ",";
        csv_file << avg_total << "," << endl;
        csv_file.close();
    }
#endif

    MPI_Finalize();

#ifdef PERF
    if(rank == 0) {
        auto main_end = chrono::high_resolution_clock::now();
        auto total_time = chrono::duration_cast<chrono::nanoseconds>(main_end - main_start).count();
        cout << "Total time: " << total_time << " ns" << endl;
        cout << "Total time: " << (total_time/1e9) << " s" << endl;
    }
#endif
    return 0;
}