#include "vlasova_a_matrix_multiply_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "util/include/util.hpp"
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"

namespace vlasova_a_matrix_multiply_ccs {

namespace {
constexpr double EPSILON = 1e-10;
}

VlasovaAMatrixMultiplySEQ::VlasovaAMatrixMultiplySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS{};
}

bool VlasovaAMatrixMultiplySEQ::ValidationImpl() {
  const auto &[A, B] = GetInput();
  return (A.rows > 0 && A.cols > 0 && B.rows > 0 && B.cols > 0 && A.cols == B.rows);
}

bool VlasovaAMatrixMultiplySEQ::PreProcessingImpl() {
  return true;
}

void VlasovaAMatrixMultiplySEQ::transposeMatrix(const SparseMatrixCCS &A, SparseMatrixCCS &AT) {
  AT.rows = A.cols;
  AT.cols = A.rows;
  AT.nnz = A.nnz;

  if (A.nnz == 0) {
    AT.values.clear();
    AT.row_indices.clear();
    AT.col_ptrs.assign(AT.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(AT.cols, 0);
  for (int i = 0; i < A.nnz; i++) {
    row_count[A.row_indices[i]]++;
  }

  AT.col_ptrs.resize(AT.cols + 1);
  AT.col_ptrs[0] = 0;
  for (int i = 0; i < AT.cols; i++) {
    AT.col_ptrs[i + 1] = AT.col_ptrs[i] + row_count[i];
  }

  AT.values.resize(A.nnz);
  AT.row_indices.resize(A.nnz);

  std::vector<int> current_pos(AT.cols, 0);
  for (int col = 0; col < A.cols; col++) {
    for (int i = A.col_ptrs[col]; i < A.col_ptrs[col + 1]; i++) {
      int row = A.row_indices[i];
      double val = A.values[i];

      int pos = AT.col_ptrs[row] + current_pos[row];
      AT.values[pos] = val;
      AT.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

void VlasovaAMatrixMultiplySEQ::multiplyMatrices(const SparseMatrixCCS &A, const SparseMatrixCCS &B,
                                                 SparseMatrixCCS &C) {
  SparseMatrixCCS AT;
  transposeMatrix(A, AT);

  C.rows = A.rows;
  C.cols = B.cols;
  C.col_ptrs.push_back(0);

  std::vector<double> temp_row(C.rows, 0.0);
  std::vector<int> row_marker(C.rows, -1);

  for (int j = 0; j < B.cols; j++) {
    for (int k = B.col_ptrs[j]; k < B.col_ptrs[j + 1]; k++) {
      int row_b = B.row_indices[k];
      double val_b = B.values[k];

      for (int l = AT.col_ptrs[row_b]; l < AT.col_ptrs[row_b + 1]; l++) {
        int row_a = AT.row_indices[l];
        double val_a = AT.values[l];

        if (row_marker[row_a] != j) {
          row_marker[row_a] = j;
          temp_row[row_a] = val_a * val_b;
        } else {
          temp_row[row_a] += val_a * val_b;
        }
      }
    }

    for (int i = 0; i < C.rows; i++) {
      if (row_marker[i] == j && std::abs(temp_row[i]) > EPSILON) {
        C.values.push_back(temp_row[i]);
        C.row_indices.push_back(i);
      }
    }
    C.col_ptrs.push_back(C.values.size());
  }

  C.nnz = C.values.size();
}

bool VlasovaAMatrixMultiplySEQ::RunImpl() {
  const auto &[A, B] = GetInput();

  try {
    SparseMatrixCCS C;
    multiplyMatrices(A, B, C);
    GetOutput() = C;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool VlasovaAMatrixMultiplySEQ::PostProcessingImpl() {
  const auto &C = GetOutput();
  return C.rows > 0 && C.cols > 0 && C.col_ptrs.size() == static_cast<size_t>(C.cols + 1);
}

}  // namespace vlasova_a_matrix_multiply_ccs
