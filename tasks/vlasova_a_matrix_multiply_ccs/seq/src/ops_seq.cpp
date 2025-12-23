#include "vlasova_a_matrix_multiply_ccs/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include "util/include/util.hpp"
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"

namespace vlasova_a_matrix_multiply_ccs {

namespace {
constexpr double kEpsilon = 1e-10;
}

VlasovaAMatrixMultiplySEQ::VlasovaAMatrixMultiplySEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS{};
}

bool VlasovaAMatrixMultiplySEQ::ValidationImpl() {
  const auto &[a, b] = GetInput();
  return (a.rows > 0 && a.cols > 0 && b.rows > 0 && b.cols > 0 && a.cols == b.rows);
}

bool VlasovaAMatrixMultiplySEQ::PreProcessingImpl() {
  return true;
}

void VlasovaAMatrixMultiplySEQ::TransposeMatrix(const SparseMatrixCCS &a, SparseMatrixCCS &at) {
  at.rows = a.cols;
  at.cols = a.rows;
  at.nnz = a.nnz;

  if (a.nnz == 0) {
    at.values.clear();
    at.row_indices.clear();
    at.col_ptrs.assign(at.cols + 1, 0);
    return;
  }

  std::vector<int> row_count(at.cols, 0);
  for (int i = 0; i < a.nnz; i++) {
    row_count[a.row_indices[i]]++;
  }

  at.col_ptrs.resize(at.cols + 1);
  at.col_ptrs[0] = 0;
  for (int i = 0; i < at.cols; i++) {
    at.col_ptrs[i + 1] = at.col_ptrs[i] + row_count[i];
  }

  at.values.resize(a.nnz);
  at.row_indices.resize(a.nnz);

  std::vector<int> current_pos(at.cols, 0);
  for (int col = 0; col < a.cols; col++) {
    for (int i = a.col_ptrs[col]; i < a.col_ptrs[col + 1]; i++) {
      int row = a.row_indices[i];
      double val = a.values[i];

      int pos = at.col_ptrs[row] + current_pos[row];
      at.values[pos] = val;
      at.row_indices[pos] = col;
      current_pos[row]++;
    }
  }
}

void VlasovaAMatrixMultiplySEQ::MultiplyMatrices(const SparseMatrixCCS &a, const SparseMatrixCCS &b,
                                                 SparseMatrixCCS &c) {
  SparseMatrixCCS at;
  TransposeMatrix(a, at);

  c.rows = a.rows;
  c.cols = b.cols;
  c.col_ptrs.push_back(0);

  std::vector<double> temp_row(c.rows, 0.0);
  std::vector<int> row_marker(c.rows, -1);

  for (int j = 0; j < b.cols; j++) {
    for (int k = b.col_ptrs[j]; k < b.col_ptrs[j + 1]; k++) {
      int row_b = b.row_indices[k];
      double val_b = b.values[k];

      for (int l = at.col_ptrs[row_b]; l < at.col_ptrs[row_b + 1]; l++) {
        int row_a = at.row_indices[l];
        double val_a = at.values[l];

        if (row_marker[row_a] != j) {
          row_marker[row_a] = j;
          temp_row[row_a] = val_a * val_b;
        } else {
          temp_row[row_a] += val_a * val_b;
        }
      }
    }

    for (int i = 0; i < c.rows; i++) {
      if (row_marker[i] == j && std::abs(temp_row[i]) > kEpsilon) {
        c.values.push_back(temp_row[i]);
        c.row_indices.push_back(i);
      }
    }
    c.col_ptrs.push_back(c.values.size());
  }

  c.nnz = c.values.size();
}

bool VlasovaAMatrixMultiplySEQ::RunImpl() {
  const auto &[a, b] = GetInput();

  try {
    SparseMatrixCCS c;
    MultiplyMatrices(a, b, c);
    GetOutput() = c;
    return true;
  } catch (const std::exception &) {
    return false;
  }
}

bool VlasovaAMatrixMultiplySEQ::PostProcessingImpl() {
  const auto &c = GetOutput();
  return c.rows > 0 && c.cols > 0 && c.col_ptrs.size() == static_cast<size_t>(c.cols + 1);
}

}  // namespace vlasova_a_matrix_multiply_ccs
