#include "oe.h"

using namespace std;

OE_sort::OE_sort(int rank, int task_num, int file_size, const char *input_file,
                 const char *output_file)
    : rank(rank), task_num(task_num), main_buffer(rank % 2 ? buffer1 : buffer0),
      input_file(input_file), output_file(output_file) {

    // Data partition
    num_per_task = file_size / task_num;
    res = file_size % task_num;

#ifdef PERF
    read_time = 0.0;
    wirte_time = 0.0;
    MPI_transmission_time = 0.0;
    MPI_sync_time = 0.0;
    merge_time = 0.0;
    glob_read = 0.0;
    glob_write = 0.0;
    glob_trans = 0.0;
    glob_merge = 0.0;
#endif

    if (num_per_task <= 500) {
        schedule = single;
        if (rank == 0) {
            size = file_size;
            offset = 0;
            left_size = 0;
            right_size = 0;
            buffer0 = new float[size];
            buffer1 = nullptr;
            neighbor_buffer = nullptr;
            std::fill_n(buffer0, size, 0);
        } else {
            size = 0;
            offset = 0;
            left_size = 0;
            right_size = 0;
            buffer0 = nullptr;
            buffer1 = nullptr;
            neighbor_buffer = nullptr;
        }
    } else {
        schedule = parallel;
        size =
            num_per_task + (rank < res); // remaining parts will divided by all
        offset = num_per_task * rank + std::min(rank, res);
        // Calculate left/right buffer size
        left_size = num_per_task + ((rank - 1) < res);
        right_size = num_per_task + ((rank + 1) < res);

        neighbor_buffer = new float[std::max(left_size, right_size)];
        // neighbor_buffer = new float[left_size];
        buffer0 = new float[size];
        buffer1 = new float[size];
        std::fill_n(neighbor_buffer, left_size, 0);
        std::fill_n(buffer0, size, 0);
        std::fill_n(buffer1, size, 0);
    }
}

OE_sort::~OE_sort() {
    if (schedule == parallel) {
        delete[](buffer0);
        delete[](buffer1);
        delete[](neighbor_buffer);
    } else if (schedule == single) {
        if (rank == 0) {
            delete[](buffer0);
        }
    }
}

void OE_sort::parallel_read_file() {
#ifdef PERF
    auto r_start = chrono::high_resolution_clock::now();
#endif

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &fh);
    MPI_File_read_at_all(fh, offset * sizeof(float), main_buffer, size,
                         MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

#ifdef PERF
    auto r_end = chrono::high_resolution_clock::now();
    read_time =
        chrono::duration_cast<chrono::nanoseconds>(r_end - r_start).count();
#endif
}

void OE_sort::single_read_file() {
#ifdef PERF
    auto r_start = chrono::high_resolution_clock::now();
#endif

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, input_file, MPI_MODE_RDONLY, MPI_INFO_NULL,
                  &fh);
    if (rank == 0) {
        MPI_File_read_at(fh, 0, buffer0, size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);

#ifdef PERF
    auto r_end = chrono::high_resolution_clock::now();
    read_time =
        chrono::duration_cast<chrono::nanoseconds>(r_end - r_start).count();
#endif
}

void OE_sort::parallel_write_file() {
#ifdef PERF
    auto w_start = chrono::high_resolution_clock::now();
#endif

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, output_file,
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    MPI_File_write_at_all(fh, offset * sizeof(float), main_buffer, size,
                          MPI_FLOAT, MPI_STATUS_IGNORE);
    MPI_File_close(&fh);

#ifdef PERF
    auto w_end = chrono::high_resolution_clock::now();
    read_time =
        chrono::duration_cast<chrono::nanoseconds>(w_end - w_start).count();
#endif
}

void OE_sort::single_write_file() {
#ifdef PERF
    auto w_start = chrono::high_resolution_clock::now();
#endif

    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, output_file,
                  MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &fh);
    if (rank == 0) {
        MPI_File_write_at(fh, 0, buffer0, size, MPI_FLOAT, MPI_STATUS_IGNORE);
    }
    MPI_File_close(&fh);

#ifdef PERF
    auto w_end = chrono::high_resolution_clock::now();
    read_time =
        chrono::duration_cast<chrono::nanoseconds>(w_end - w_start).count();
#endif
}

void OE_sort::parallel_sort() {
    bool global_sorted(false);
    bool local_sorted(false);

    // use STL to sort local content
    std::sort(main_buffer, main_buffer + size);
    // Split odd rank & even rank
    if (rank % 2 == 1) {
        while (not global_sorted) {
            local_sorted = true;
            bool local_sorted_1 = not _do_left();
            bool local_sorted_2 = not _do_right();
            local_sorted = (local_sorted_1 & local_sorted_2);
            // Sync sorting status
            MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);
        }
    } else {
        while (not global_sorted) {
            local_sorted = true;
            bool local_sorted_1 = not _do_right();
            bool local_sorted_2 = not _do_left();
            local_sorted = (local_sorted_1 & local_sorted_2);
            // Sync sorting status
            MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);
        }
    }
}

void OE_sort::single_sort() {
    if (rank == 0) {
        std::sort(buffer0, buffer0 + size);
    }
}

bool OE_sort::_do_left() {
    if (rank == 0 || size == 0) {
        std::swap(buffer0, buffer1);
        return false;
    }
    MPI_Request request;

#ifdef PERF
    auto mpi_start = chrono::high_resolution_clock::now();
#endif

    MPI_Irecv(neighbor_buffer, left_size, MPI_FLOAT, rank - 1, mpi_tags::right,
              MPI_COMM_WORLD, &request);
    MPI_Send(buffer1, size, MPI_FLOAT, rank - 1, mpi_tags::left,
             MPI_COMM_WORLD);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

#ifdef PERF
    auto mpi_end = chrono::high_resolution_clock::now();
    MPI_transmission_time +=
        chrono::duration_cast<chrono::nanoseconds>(mpi_end - mpi_start).count();
#endif

    if (neighbor_buffer[left_size - 1] <= buffer1[0]) {
        std::swap(buffer0, buffer1);
        return false;
    }

#ifdef PERF
    auto merge_start = chrono::high_resolution_clock::now();
#endif

    _merge_large(buffer1, size, neighbor_buffer, left_size, buffer0, size);

#ifdef PERF
    auto merge_end = chrono::high_resolution_clock::now();
    merge_time +=
        chrono::duration_cast<chrono::nanoseconds>(merge_end - merge_start)
            .count();
#endif

    return true;
}

bool OE_sort::_do_right() {
    if ((rank + 1) == task_num || size == 0 || right_size == 0) {
        std::swap(buffer0, buffer1);
        return false;
    }
    MPI_Request request;

#ifdef PERF
    auto mpi_start = chrono::high_resolution_clock::now();
#endif

    MPI_Irecv(neighbor_buffer, right_size, MPI_FLOAT, rank + 1, mpi_tags::left,
              MPI_COMM_WORLD, &request);
    MPI_Send(buffer0, size, MPI_FLOAT, rank + 1, mpi_tags::right,
             MPI_COMM_WORLD);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

#ifdef PERF
    auto mpi_end = chrono::high_resolution_clock::now();
    MPI_transmission_time +=
        chrono::duration_cast<chrono::nanoseconds>(mpi_end - mpi_start).count();
#endif

    if (buffer0[size - 1] <= neighbor_buffer[0]) {
        std::swap(buffer0, buffer1);
        return false;
    }

#ifdef PERF
    auto merge_start = chrono::high_resolution_clock::now();
#endif

    _merge_small(buffer0, size, neighbor_buffer, right_size, buffer1, size);

#ifdef PERF
    auto merge_end = chrono::high_resolution_clock::now();
    merge_time +=
        chrono::duration_cast<chrono::nanoseconds>(merge_end - merge_start)
            .count();
#endif

    return true;
}

void OE_sort::_merge_small(float *src1, ssize_t src1_size, float *src2,
                           ssize_t src2_size, float *dest, ssize_t dest_size) {

    const float *src1_end(src1 + src1_size), *src2_end(src2 + src2_size),
        *dest_end(dest + dest_size);
    float *src1_iter(src1), *src2_iter(src2), *dest_iter(dest);

    // Fill data
    while (src1_iter < src1_end && src2_iter < src2_end &&
           dest_iter < dest_end) {
        if (*src1_iter < *src2_iter) {
            *dest_iter++ = *src1_iter++;
        } else {
            *dest_iter++ = *src2_iter++;
        }
    }

    // Fill blanks in dest
    if (dest_iter < dest_end) {
        int offset(0);
        if (src1_iter < src1_end) {
            offset = std::min(dest_end - dest_iter, src1_end - src1_iter);
            std::copy(src1_iter, src1_iter + offset, dest_iter);
        } else {
            offset = std::min(dest_end - dest_iter, src2_end - src2_iter);
            std::copy(src2_iter, src2_iter + offset, dest_iter);
        }
    }
}

void OE_sort::_merge_large(float *src1, ssize_t src1_size, float *src2,
                           ssize_t src2_size, float *dest, ssize_t dest_size) {

    const float *src1_end(src1 + src1_size), *src2_end(src2 + src2_size),
        *dest_end(dest + dest_size);
    float *src1_iter(const_cast<float *>(src1_end) - 1),
        *src2_iter(const_cast<float *>(src2_end) - 1),
        *dest_iter(const_cast<float *>(dest_end) - 1);

    // Fill data
    while (src1 <= src1_iter && src2 <= src2_iter && dest <= dest_iter) {
        if (*src1_iter > *src2_iter) {
            *dest_iter-- = *src1_iter--;
        } else {
            *dest_iter-- = *src2_iter--;
        }
    }

    // Fill blanks in dest
    if (dest <= dest_iter) {
        int offset(0);
        if (src1_iter <= src1_end) {
            offset = std::min(dest_iter - dest, src1_iter - src1) + 1;
            std::copy(src1_iter, src1_iter + offset, dest_iter);
        } else {
            offset = std::min(dest_iter - dest, src2_iter - src2) + 1;
            std::copy(src2_iter, src2_iter + offset, dest_iter);
        }
    }
}