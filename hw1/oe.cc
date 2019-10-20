#include "oe.h"

using namespace std;

OE_sort::OE_sort(int rank, int task_num, int file_size, const char *input_file,
                 const char *output_file)
    : rank(rank), task_num(task_num),
      main_buffer(rank % 2 ? odd_buffer : even_buffer), input_file(input_file),
      output_file(output_file) {

    // Data partition
    num_per_task = file_size / task_num;
    res = file_size % task_num;
    size = num_per_task + (rank < res); // remaining parts will divided by all
    offset = num_per_task * rank + std::min(rank, res);

    left_size = num_per_task + ((rank - 1) < res);
    right_size = num_per_task + ((rank + 1) < res);

#ifdef PERF
    auto mem_start = chrono::high_resolution_clock::now();
#endif

    if (left_size > 0) {
        neighbor_buffer = new float[left_size];
    } else {
        neighbor_buffer = nullptr;
    }
    if (size > 0) {
        even_buffer = new float[size];
        odd_buffer = new float[size];
    } else {
        even_buffer = nullptr;
        odd_buffer = nullptr;
    }

#ifdef PERF
    auto mem_end = chrono::high_resolution_clock::now();
    mem_time =
        chrono::duration_cast<chrono::nanoseconds>(mem_end - mem_start).count();
#endif

}

OE_sort::~OE_sort() {
    delete[](even_buffer);
    delete[](odd_buffer);
    delete[](neighbor_buffer);
}

void OE_sort::read_file() {
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

void OE_sort::write_file() {
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
    write_time =
        chrono::duration_cast<chrono::nanoseconds>(w_end - w_start).count();
#endif

}

void OE_sort::sort() {
    bool global_sorted(false);
    bool local_sorted(false);

#ifdef PERF
    auto stl_start = chrono::high_resolution_clock::now();
#endif

    // use Boost to sort local content
    boost::sort::spreadsort::float_sort(main_buffer, main_buffer + size);

#ifdef PERF
    auto stl_end = chrono::high_resolution_clock::now();
    stl_sort_time +=
        chrono::duration_cast<chrono::nanoseconds>(stl_end - stl_start).count();
#endif

    // Split odd rank & even rank
    if (rank % 2 == 1) {
        while (not global_sorted) {
            local_sorted = true;
            local_sorted = _left() & _right();

#ifdef PERF
            auto sync_start = chrono::high_resolution_clock::now();
#endif

            // Sync sorting status
            MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);

#ifdef PERF
            auto sync_end = chrono::high_resolution_clock::now();
            MPI_sync_time += chrono::duration_cast<chrono::nanoseconds>(
                                 sync_end - sync_start)
                                 .count();
#endif

        }
    } else {
        while (not global_sorted) {
            local_sorted = true;
            local_sorted = _right() & _left();

#ifdef PERF
            auto sync_start = chrono::high_resolution_clock::now();
#endif

            // Sync sorting status
            MPI_Allreduce(&local_sorted, &global_sorted, 1, MPI::BOOL, MPI_LAND,
                          MPI_COMM_WORLD);

#ifdef PERF
            auto sync_end = chrono::high_resolution_clock::now();
            MPI_sync_time += chrono::duration_cast<chrono::nanoseconds>(
                                 sync_end - sync_start)
                                 .count();
#endif

        }
    }
}

bool OE_sort::_left() {
    if (rank == 0 || size == 0) {
        std::swap(even_buffer, odd_buffer);
        return true;
    }

#ifdef PERF
    auto mpi_start = chrono::high_resolution_clock::now();
#endif

    MPI_Irecv(neighbor_buffer, left_size, MPI_FLOAT, rank - 1, mpi_tags::right,
              MPI_COMM_WORLD, &request);
    MPI_Send(odd_buffer, size, MPI_FLOAT, rank - 1, mpi_tags::left,
             MPI_COMM_WORLD);
    MPI_Wait(&request, MPI_STATUS_IGNORE);

#ifdef PERF
    auto mpi_end = chrono::high_resolution_clock::now();
    MPI_transmission_time +=
        chrono::duration_cast<chrono::nanoseconds>(mpi_end - mpi_start).count();
#endif

    if (neighbor_buffer[left_size - 1] <= odd_buffer[0]) {
        std::swap(even_buffer, odd_buffer);
        return true;
    }
    _merge_large(odd_buffer, size, neighbor_buffer, left_size, even_buffer,
                 size);
    return false;
}

bool OE_sort::_right() {
    if ((rank + 1) == task_num || size == 0 || right_size == 0) {
        std::swap(even_buffer, odd_buffer);
        return true;
    }

#ifdef PERF
    auto mpi_start = chrono::high_resolution_clock::now();
#endif

    MPI_Send(even_buffer, size, MPI_FLOAT, rank + 1, mpi_tags::right,
             MPI_COMM_WORLD);
    MPI_Irecv(neighbor_buffer, right_size, MPI_FLOAT, rank + 1, mpi_tags::left,
              MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
    if (even_buffer[size - 1] <= neighbor_buffer[0]) {
        std::swap(even_buffer, odd_buffer);
        return true;
    }

#ifdef PERF
    auto mpi_end = chrono::high_resolution_clock::now();
    MPI_transmission_time +=
        chrono::duration_cast<chrono::nanoseconds>(mpi_end - mpi_start).count();
#endif

    _merge_small(even_buffer, size, neighbor_buffer, right_size, odd_buffer,
                 size);
    return false;
}

void OE_sort::_merge_small(float *src1, size_t src1_size, float *src2,
                           size_t src2_size, float *dest, size_t dest_size) {

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

void OE_sort::_merge_large(float *src1, size_t src1_size, float *src2,
                           size_t src2_size, float *dest, size_t dest_size) {

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