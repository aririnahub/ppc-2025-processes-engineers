#pragma once

#include <string>
#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace vlasova_a_matrix_multiply_ccs {

struct SparseMatrixCCS {
  std::vector<double> values;
  std::vector<int> row_indices;
  std::vector<int> col_ptrs;
  int rows;
  int cols;
  int nnz;
};

using InType = std::pair<SparseMatrixCCS, SparseMatrixCCS>;
using OutType = SparseMatrixCCS;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

SparseMatrixCCS generateRandomSparseMatrix(int rows, int cols, double density);
bool compareMatrices(const SparseMatrixCCS &A, const SparseMatrixCCS &B, double epsilon = 1e-6);

void transposeMatrixCCS(const SparseMatrixCCS &A, SparseMatrixCCS &AT);

}  // namespace vlasova_a_matrix_multiply_ccs
