#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"
#include "util/include/util.hpp"

namespace vlasova_a_elem_matrix_sum{

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
  for (int i = 0; i < GetInput().size(); ++i) {
    int row_sum = 0;
    for (int j = 0; j < GetInput()[i].size(); ++j) {
      row_sum += GetInput()[i][j];
    }
    GetOutput()[i] = row_sum;
  }
  return true;
}

bool VlasovaAElemMatrixSumSEQ::PostProcessingImpl() {
  return !GetOutput().empty();
}

}  // namespace vlasova_a_elem_matrix_sum
