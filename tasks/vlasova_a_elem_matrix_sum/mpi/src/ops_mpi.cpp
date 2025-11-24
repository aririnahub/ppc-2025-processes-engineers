#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

VlasovaAElemMatrixSumMPI::VlasovaAElemMatrixSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool VlasovaAElemMatrixSumMPI::ValidationImpl() {
  return GetOutput().empty();
}

bool VlasovaAElemMatrixSumMPI::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VlasovaAElemMatrixSumMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const size_t total_rows = GetInput().size();
  if (total_rows == 0) {
    return true;
  }

  const size_t base_chunk = total_rows / size;
  const size_t remainder = total_rows % size;

  const auto rank_size = static_cast<size_t>(rank);
  const size_t local_rows = base_chunk + (rank_size < remainder ? 1 : 0);
  const size_t start_row = (rank_size * base_chunk) + std::min<size_t>(rank_size, remainder);

  std::vector<int> local_sums(local_rows);
  for (size_t i = 0; i < local_rows; ++i) {
    const size_t global_row = start_row + i;
    int row_sum = 0;
    for (int val : GetInput()[global_row]) {
      row_sum += val;
    }
    local_sums[i] = row_sum;
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);

  size_t current_displ = 0;
  for (int proc = 0; proc < size; ++proc) {
    const auto proc_size = static_cast<size_t>(proc);
    const size_t proc_rows = base_chunk + (proc_size < remainder ? 1 : 0);
    recv_counts[proc] = static_cast<int>(proc_rows);
    displs[proc] = static_cast<int>(current_displ);
    current_displ += proc_rows;
  }

  MPI_Allgatherv(local_sums.data(), static_cast<int>(local_rows), MPI_INT, GetOutput().data(), recv_counts.data(),
                 displs.data(), MPI_INT, MPI_COMM_WORLD);

  return true;
}

bool VlasovaAElemMatrixSumMPI::PostProcessingImpl() {
  return true;
}

}  // namespace vlasova_a_elem_matrix_sum
