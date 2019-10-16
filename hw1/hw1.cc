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
        // cout << "AVG Mem: " << global_results[0] << " ns" << endl;
        // cout << "AVG Read: " << global_results[1] << " ns" << endl;
        // cout << "AVG Write: " << global_results[2] << " ns" << endl;
        // cout << "AVG MPI Trans: " << global_results[3] << " ns" << endl;
        // cout << "AVG MPI Sync: " << global_results[4] << " ns" << endl;
        // cout << "AVG Merge: " << global_results[5] << " ns" << endl;
        // cout << "AVG STL sort: " << global_results[6] << " ns" << endl;
        // auto avg_total = std::accumulate(global_results, global_results + 7, 0.0) / 1e9;
        // cout << "Avg total: " << avg_total << " s" << endl;
        ofstream file;
        string filename(argv[3]);
#ifdef PARA_STL
        filename.append("_para_stl");
#endif
        filename.append("_perf.txt");
        size_t pos = filename.find("./");
        if (pos != std::string::npos) {
            // If found then erase it from string
            filename.erase(pos, 2);
        }
        file.open(filename.c_str());
        file << "AVG Mem: " << global_results[0] << " ns" << endl;
        file << "AVG Read: " << global_results[1] << " ns" << endl;
        file << "AVG Write: " << global_results[2] << " ns" << endl;
        file << "AVG MPI Trans: " << global_results[3] << " ns" << endl;
        file << "AVG MPI Sync: " << global_results[4] << " ns" << endl;
        file << "AVG Merge: " << global_results[5] << " ns" << endl;
        file << "AVG STL sort: " << global_results[6] << " ns" << endl;
        auto avg_total = std::accumulate(global_results, global_results + 7, 0.0) / 1e9;
        file << "Avg total: " << avg_total << " s" << endl;
        file.close();
    }
#endif

    MPI_Finalize();
    return 0;
}