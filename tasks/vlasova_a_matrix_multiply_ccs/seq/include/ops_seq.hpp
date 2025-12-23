#pragma once

#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"
#include "task/include/task.hpp"

namespace vlasova_a_matrix_multiply_ccs {
  class VlasovaAMatrixMultiplySEQ : public BaseTask {
  public:
    static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
      return ppc::task::TypeOfTask::kSEQ;
    }
    explicit VlasovaAMatrixMultiplySEQ(const InType &in);

  private:
    bool ValidationImpl() override;
    bool PreProcessingImpl() override;
    bool RunImpl() override;
    bool PostProcessingImpl() override;

    void transposeMatrix(const SparseMatrixCCS& A, SparseMatrixCCS& AT);
    void multiplyMatrices(const SparseMatrixCCS& A, const SparseMatrixCCS& B, SparseMatrixCCS& C);
  };

}  // namespace vlasova_a_matrix_multiply_ccs