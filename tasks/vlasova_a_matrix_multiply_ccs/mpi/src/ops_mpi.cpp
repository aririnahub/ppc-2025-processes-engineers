#include "vlasova_a_matrix_multiply_ccs/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

#include "util/include/util.hpp"
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"

namespace vlasova_a_matrix_multiply_ccs {

namespace {
constexpr double EPSILON = 1e-10;
}

VlasovaAMatrixMultiplyMPI::VlasovaAMatrixMultiplyMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = SparseMatrixCCS{};
}

bool VlasovaAMatrixMultiplyMPI::ValidationImpl() {
  const auto &[A, B] = GetInput();
  return (A.rows > 0 && A.cols > 0 && B.rows > 0 && B.cols > 0 && A.cols == B.rows);
}

bool VlasovaAMatrixMultiplyMPI::PreProcessingImpl() {
  return true;
}

void VlasovaAMatrixMultiplyMPI::transposeMatrixMPI(const SparseMatrixCCS &A, SparseMatrixCCS &AT) {
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

std::pair<int, int> VlasovaAMatrixMultiplyMPI::splitColumns(int total_cols, int rank, int size) {
  int base_cols = total_cols / size;
  int remainder = total_cols % size;

  int start_col = rank * base_cols + std::min(rank, remainder);
  int end_col = start_col + base_cols + (rank < remainder ? 1 : 0);

  return {start_col, end_col};
}

void VlasovaAMatrixMultiplyMPI::extractLocalColumns(const SparseMatrixCCS &B, int start_col, int end_col,
                                                    std::vector<double> &loc_val, std::vector<int> &loc_row_ind,
                                                    std::vector<int> &loc_col_ptr) {
  loc_val.clear();
  loc_row_ind.clear();
  loc_col_ptr.clear();

  loc_col_ptr.push_back(0);

  for (int col = start_col; col < end_col; col++) {
    int start_idx = B.col_ptrs[col];
    int end_idx = B.col_ptrs[col + 1];

    for (int i = start_idx; i < end_idx; i++) {
      loc_val.push_back(B.values[i]);
      loc_row_ind.push_back(B.row_indices[i]);
    }

    loc_col_ptr.push_back(loc_val.size());
  }
}

void VlasovaAMatrixMultiplyMPI::multiplyLocalMatrices(const SparseMatrixCCS &AT, const std::vector<double> &loc_val,
                                                      const std::vector<int> &loc_row_ind,
                                                      const std::vector<int> &loc_col_ptr, int loc_cols,
                                                      std::vector<double> &res_val, std::vector<int> &res_row_ind,
                                                      std::vector<int> &res_col_ptr) {
  res_val.clear();
  res_row_ind.clear();
  res_col_ptr.clear();
  res_col_ptr.push_back(0);

  std::vector<double> temp_row(AT.cols, 0.0);
  std::vector<int> row_marker(AT.cols, -1);

  for (int j = 0; j < loc_cols; j++) {
    int col_start = loc_col_ptr[j];
    int col_end = loc_col_ptr[j + 1];

    for (int k = col_start; k < col_end; k++) {
      int row_b = loc_row_ind[k];
      double val_b = loc_val[k];

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

    for (int i = 0; i < AT.cols; i++) {
      if (row_marker[i] == j && std::abs(temp_row[i]) > EPSILON) {
        res_val.push_back(temp_row[i]);
        res_row_ind.push_back(i);
      }
    }
    res_col_ptr.push_back(res_val.size());
  }
}

bool VlasovaAMatrixMultiplyMPI::RunImpl() {
  const auto &[A, B] = GetInput();

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  SparseMatrixCCS AT;
  if (rank == 0) {
    transposeMatrixMPI(A, AT);
  } else {
    AT.rows = A.cols;
    AT.cols = A.rows;
  }

  MPI_Bcast(&AT.rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&AT.cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    AT.nnz = AT.values.size();
  }
  MPI_Bcast(&AT.nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank != 0) {
    AT.values.resize(AT.nnz);
    AT.row_indices.resize(AT.nnz);
    AT.col_ptrs.resize(AT.cols + 1);
  }

  MPI_Bcast(AT.values.data(), AT.nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(AT.row_indices.data(), AT.nnz, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(AT.col_ptrs.data(), AT.cols + 1, MPI_INT, 0, MPI_COMM_WORLD);

  auto [start_col, end_col] = splitColumns(B.cols, rank, size);
  int loc_cols = end_col - start_col;

  std::vector<double> loc_B_val;
  std::vector<int> loc_B_row_ind;
  std::vector<int> loc_B_col_ptr;

  extractLocalColumns(B, start_col, end_col, loc_B_val, loc_B_row_ind, loc_B_col_ptr);

  std::vector<double> loc_res_val;
  std::vector<int> loc_res_row_ind;
  std::vector<int> loc_res_col_ptr;

  multiplyLocalMatrices(AT, loc_B_val, loc_B_row_ind, loc_B_col_ptr, loc_cols, loc_res_val, loc_res_row_ind,
                        loc_res_col_ptr);

  if (rank == 0) {
    SparseMatrixCCS C;
    C.rows = A.rows;
    C.cols = B.cols;

    std::vector<std::vector<double>> all_values(size);
    std::vector<std::vector<int>> all_row_indices(size);
    std::vector<std::vector<int>> all_col_ptrs(size);

    all_values[0] = loc_res_val;
    all_row_indices[0] = loc_res_row_ind;
    all_col_ptrs[0] = loc_res_col_ptr;

    for (int src = 1; src < size; src++) {
      int src_nnz, src_cols;
      MPI_Recv(&src_nnz, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(&src_cols, 1, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      std::vector<double> src_vals(src_nnz);
      std::vector<int> src_rows(src_nnz);
      std::vector<int> src_ptrs(src_cols + 1);

      MPI_Recv(src_vals.data(), src_nnz, MPI_DOUBLE, src, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(src_rows.data(), src_nnz, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      MPI_Recv(src_ptrs.data(), src_cols + 1, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      all_values[src] = std::move(src_vals);
      all_row_indices[src] = std::move(src_rows);
      all_col_ptrs[src] = std::move(src_ptrs);
    }

    C.col_ptrs.push_back(0);

    std::vector<int> value_offsets(size, 0);
    std::vector<int> col_offsets(size, 0);

    for (int i = 0; i < size; i++) {
      if (i > 0) {
        value_offsets[i] = value_offsets[i - 1] + all_values[i - 1].size();
        col_offsets[i] = col_offsets[i - 1] + (all_col_ptrs[i - 1].size() - 1);
      }
    }

    for (int i = 0; i < size; i++) {
      C.values.insert(C.values.end(), all_values[i].begin(), all_values[i].end());
      C.row_indices.insert(C.row_indices.end(), all_row_indices[i].begin(), all_row_indices[i].end());

      for (size_t j = 1; j < all_col_ptrs[i].size(); j++) {
        C.col_ptrs.push_back(all_col_ptrs[i][j] + value_offsets[i]);
      }
    }

    C.nnz = C.values.size();
    GetOutput() = C;

  } else {
    int local_nnz = loc_res_val.size();
    int local_cols = loc_cols;

    MPI_Send(&local_nnz, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    MPI_Send(&local_cols, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    MPI_Send(loc_res_val.data(), local_nnz, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD);
    MPI_Send(loc_res_row_ind.data(), local_nnz, MPI_INT, 0, 3, MPI_COMM_WORLD);
    MPI_Send(loc_res_col_ptr.data(), loc_cols + 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool VlasovaAMatrixMultiplyMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    const auto &C = GetOutput();
    return C.rows > 0 && C.cols > 0 && C.col_ptrs.size() == static_cast<size_t>(C.cols + 1);
  } else {
    const auto &C = GetOutput();
    return C.rows == 0 && C.cols == 0;
  }
}

}  // namespace vlasova_a_matrix_multiply_ccs
