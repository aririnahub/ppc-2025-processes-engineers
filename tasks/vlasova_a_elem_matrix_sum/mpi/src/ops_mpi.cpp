#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

VlasovaAElemMatrixSumMPI::VlasovaAElemMatrixSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VlasovaAElemMatrixSumMPI::ValidationImpl() {
  return !GetInput().empty();
}

bool VlasovaAElemMatrixSumMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool VlasovaAElemMatrixSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t total_rows = GetInput().size();

  size_t rows_per_process = total_rows / size;
  int remainder = static_cast<int>(total_rows % size);

  size_t start = (rank * rows_per_process) + std::min(rank, remainder);
  size_t end = start + rows_per_process + (rank < remainder ? 1 : 0);
  size_t local_size = end - start;

  std::vector<int> local_results(local_size);
  for (size_t i = start; i < end; i++) {
    int row_sum = 0;
    const auto &row = GetInput()[i];
    for (int val : row) {
      row_sum += val;
    }
    local_results[i - start] = row_sum;
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  int current_displ = 0;
  for (int proc = 0; proc < size; proc++) {
    size_t proc_start = (proc * rows_per_process) + std::min(proc, remainder);
    size_t proc_end = proc_start + rows_per_process + (proc < remainder ? 1 : 0);
    recv_counts[proc] = static_cast<int>(proc_end - proc_start);
    displs[proc] = current_displ;
    current_displ += recv_counts[proc];
  }

  MPI_Allgatherv(local_results.data(), static_cast<int>(local_size), MPI_INT, GetOutput().data(), recv_counts.data(),
                 displs.data(), MPI_INT, MPI_COMM_WORLD);

  return true;
}

bool VlasovaAElemMatrixSumMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace vlasova_a_elem_matrix_sum
