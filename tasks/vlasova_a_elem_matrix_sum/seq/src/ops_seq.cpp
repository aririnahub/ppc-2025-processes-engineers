#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cstddef>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"

namespace vlasova_a_elem_matrix_sum {

VlasovaAElemMatrixSumSEQ::VlasovaAElemMatrixSumSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  InType tmp(in);
  GetInput().swap(tmp);
}

bool VlasovaAElemMatrixSumSEQ::ValidationImpl() {
  return GetOutput().empty();
}

bool VlasovaAElemMatrixSumSEQ::PreProcessingImpl() {
  GetOutput().clear();
  return true;
}

bool VlasovaAElemMatrixSumSEQ::RunImpl() {
  if (GetInput().empty()) {
    return true;
  }
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
  return true;
}

}  // namespace vlasova_a_elem_matrix_sum
