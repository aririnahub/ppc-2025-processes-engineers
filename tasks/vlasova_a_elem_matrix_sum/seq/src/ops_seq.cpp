#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"

#include <cstddef>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

VlasovaAElemMatrixSumSEQ::VlasovaAElemMatrixSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // GetOutput();
}

bool VlasovaAElemMatrixSumSEQ::ValidationImpl() {
  return !GetInput().empty();
}

bool VlasovaAElemMatrixSumSEQ::PreProcessingImpl() {
  GetOutput().resize(GetInput().size());
  return true;
}

bool VlasovaAElemMatrixSumSEQ::RunImpl() {
  for (size_t i = 0; i < GetInput().size(); ++i) {
    int row_sum = 0;
    for (int val : GetInput()[i]) {
      row_sum += val;
    }
    GetOutput()[i] = row_sum;
  }
  return true;
}

bool VlasovaAElemMatrixSumSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace vlasova_a_elem_matrix_sum
