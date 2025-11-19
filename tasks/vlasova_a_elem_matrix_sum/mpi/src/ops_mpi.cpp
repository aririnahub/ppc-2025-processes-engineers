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

  int count = GetInput().size();
  int rows_per_process = count/size;
  int remainder = count % size;

  int start = rank * rows_per_process + std::min(rank, remainder);
  int end = start + rows_per_process + (rank < remainder ? 1 : 0);

  for (int i = start; i < end; i++){
    int row_sum = 0;
    for (int j = 0; j < GetInput()[i].size(); ++j){
      row_sum += GetInput()[i][j];
    }
    GetOutput()[i] = row_sum;
  }

  if (rank == 0){
    for (int proc = 1; proc < size; proc++){
      int p_start = proc * rows_per_process + std::min(proc, remainder);
      int p_end = p_start + rows_per_process + (proc < remainder ? 1 : 0);
      
      std::vector<int> p_res(p_end - p_start);
      MPI_Recv(p_res.data(), p_res.size(), MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i< p_res.size(); ++i){
        GetOutput()[p_start+i] = p_res[i];
      }
    }
  }
  else{
    std::vector<int> cur_res(end - start);
    for (int i = 0; i < cur_res.size(); ++i){
      cur_res[i] = GetOutput()[start + i];
    }

    MPI_Send(cur_res.data(), cur_res.size(), MPI_INT, 0, 0, MPI_COMM_WORLD); 
  }

  return true;
}

bool VlasovaAElemMatrixSumMPI::PostProcessingImpl(){
  return !GetOutput().empty();
}

}  // namespace vlasova_a_elem_matrix_sum