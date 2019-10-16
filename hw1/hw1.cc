// CS5540 Homework 1: Odd-Even Sort
// Author: Jerry ZJ
// Date: 2019/09/27
#include "oe.h"
#include <cassert>
using namespace std;

int main(int argc, char **argv) {

    assert(argc == 4); // Make sure arguments are ready

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
    MPI_Reduce(&oe.mem_time, &oe.avg_mem, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&oe.read_time, &oe.avg_read, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&oe.write_time, &oe.avg_write, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&oe.MPI_transmission_time, &oe.avg_trans, 1, MPI_DOUBLE, MPI_SUM,
               0, MPI_COMM_WORLD);
    MPI_Reduce(&oe.MPI_sync_time, &oe.avg_sync, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&oe.merge_time, &oe.avg_merge, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&oe.stl_sort_time, &oe.avg_stl_sort, 1, MPI_DOUBLE, MPI_SUM, 0,
               MPI_COMM_WORLD);

    // cout << "Rank: " << rank << " read: " << oe.read_time << " ns" << endl;
    // cout << "Rank: " << rank << " write: " << oe.write_time << " ns" << endl;
    if (rank == 0) {
        if (oe.schedule == parallel) {
            oe.avg_mem /= task_num;
            oe.avg_read /= task_num;
            oe.avg_write /= task_num;
            oe.avg_trans /= task_num;
            oe.avg_sync /= task_num;
            oe.avg_merge /= task_num;
            oe.avg_stl_sort /= task_num;
        }
        cout << "AVG Mem: " << oe.avg_mem << " ns" << endl;
        cout << "AVG Read: " << oe.avg_read << " ns" << endl;
        cout << "AVG Write: " << oe.avg_write << " ns" << endl;
        cout << "AVG MPI Trans: " << oe.avg_trans << " ns" << endl;
        cout << "AVG MPI Sync: " << oe.avg_sync << " ns" << endl;
        cout << "AVG Merge: " << oe.avg_merge << " ns" << endl;
        cout << "AVG STL sort: " << oe.avg_stl_sort << " ns" << endl;
        auto avg_total = (oe.avg_mem + oe.avg_read + oe.avg_write +
                          oe.avg_trans + oe.avg_merge + oe.avg_stl_sort) /
                         1e9;
        cout << "Avg total:" << avg_total << " s" << endl;
    }
#endif

    MPI_Finalize();
    return 0;
}