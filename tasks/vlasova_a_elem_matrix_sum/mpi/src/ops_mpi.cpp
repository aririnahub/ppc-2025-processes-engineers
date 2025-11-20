#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace vlasova_a_elem_matrix_sum{

VlasovaAElemMatrixSumMPI::VlasovaAElemMatrixSumMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool VlasovaAElemMatrixSumMPI::ValidationImpl() {
  return !GetInput().empty() && !GetInput()[0].empty();
}

bool VlasovaAElemMatrixSumMPI::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool VlasovaAElemMatrixSumMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int total_rows = GetInput().size();

  int rows_per_process = total_rows / size;
  int remainder = total_rows % size;
  
  int start = rank * rows_per_process + std::min(rank, remainder);
  int end = start + rows_per_process + (rank < remainder ? 1 : 0);
  int local_size = end - start;

  std::vector<int> local_results(local_size);
  for (int i = start; i < end; i++) {
    int row_sum = 0;
    const auto& row = GetInput()[i];
    for (int val : row) {
      row_sum += val;
    }
    local_results[i - start] = row_sum;
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displs(size);
  
  int current_displ = 0;
  for (int proc = 0; proc < size; proc++) {
    int proc_start = proc * rows_per_process + std::min(proc, remainder);
    int proc_end = proc_start + rows_per_process + (proc < remainder ? 1 : 0);
    recv_counts[proc] = proc_end - proc_start;
    displs[proc] = current_displ; 
    current_displ += recv_counts[proc];
  }

  MPI_Allgatherv(local_results.data(), local_size, MPI_INT,
                 GetOutput().data(), recv_counts.data(), displs.data(), MPI_INT,
                 MPI_COMM_WORLD);

  return true;
}

bool VlasovaAElemMatrixSumMPI::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace vlasova_a_elem_matrix_sum