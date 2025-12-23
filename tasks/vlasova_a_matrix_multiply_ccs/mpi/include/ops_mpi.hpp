#pragma once

#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace vlasova_a_matrix_multiply_ccs {

  class VlasovaAMatrixMultiplyMPI : public BaseTask {
  public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kMPI;
    }
    explicit VlasovaAMatrixMultiplyMPI(const InType &in);

  private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;
  
    void transposeMatrixMPI(const SparseMatrixCCS& A, SparseMatrixCCS& AT);
    std::pair<int, int> splitColumns(int total_cols, int rank, int size);
    void extractLocalColumns(const SparseMatrixCCS& B, int start_col, int end_col,
                            std::vector<double>& loc_val, 
                            std::vector<int>& loc_row_ind,
                            std::vector<int>& loc_col_ptr);
    void multiplyLocalMatrices(const SparseMatrixCCS& AT,
                              const std::vector<double>& loc_val,
                              const std::vector<int>& loc_row_ind,
                              const std::vector<int>& loc_col_ptr,
                              int loc_cols,
                              std::vector<double>& res_val,
                              std::vector<int>& res_row_ind,
                              std::vector<int>& res_col_ptr);
  };

}  // namespace vlasova_a_matrix_multiply_ccs