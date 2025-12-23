#include <gtest/gtest.h>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"
#include "vlasova_a_matrix_multiply_ccs/mpi/include/ops_mpi.hpp"
#include "vlasova_a_matrix_multiply_ccs/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace vlasova_a_matrix_multiply_ccs {

  SparseMatrixCCS generateRandomSparseMatrix(int rows, int cols, double density) {
      SparseMatrixCCS matrix;
      matrix.rows = rows;
      matrix.cols = cols;
      matrix.col_ptrs.resize(cols + 1, 0);
      
      static bool initialized = false;
      if (!initialized) {
          std::srand(static_cast<unsigned int>(std::time(nullptr)));
          initialized = true;
      }
      
      std::vector<int> col_counts(cols, 0);
      std::vector<std::vector<double>> col_values(cols);
      std::vector<std::vector<int>> col_rows(cols);
      
      int total_nnz = 0;
      
      for (int col = 0; col < cols; col++) {
          for (int row = 0; row < rows; row++) {
              double random_value = static_cast<double>(std::rand()) / RAND_MAX;
              
              if (random_value < density) {
                  double value = 0.1 + static_cast<double>(std::rand()) / RAND_MAX * 9.9;
                  
                  col_values[col].push_back(value);
                  col_rows[col].push_back(row);
                  col_counts[col]++;
                  total_nnz++;
              }
          }
      }
      
      matrix.nnz = total_nnz;
      matrix.values.resize(total_nnz);
      matrix.row_indices.resize(total_nnz);
      
      int current_idx = 0;
      matrix.col_ptrs[0] = 0;
      
      for (int col = 0; col < cols; col++) {
          for (int i = 0; i < col_counts[col]; i++) {
              matrix.values[current_idx] = col_values[col][i];
              matrix.row_indices[current_idx] = col_rows[col][i];
              current_idx++;
          }
          matrix.col_ptrs[col + 1] = current_idx;
      }
      
      return matrix;
  }

  class VlasovaMatrixMultiplyPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  protected:
    void SetUp() override {
      std::srand(static_cast<unsigned int>(std::time(nullptr)));

      int size = 100;
      double density = 0.1;
      A_ = generateRandomSparseMatrix(size, size, density);
      B_ = generateRandomSparseMatrix(size, size, density);
      input_data_ = std::make_pair(A_, B_);
    }

    bool CheckTestOutputData(OutType &output_data) final {
      if (output_data.rows == 0 && output_data.cols == 0) {
          return true;  
      }
      
      return output_data.rows == A_.rows && 
            output_data.cols == B_.cols &&
            output_data.col_ptrs.size() == static_cast<size_t>(output_data.cols + 1);
    }

    InType GetTestInputData() final {
      return input_data_;
    }

  private:
    SparseMatrixCCS A_, B_;
    InType input_data_;
  };

  TEST_P(VlasovaMatrixMultiplyPerfTest, RunPerfModes) {
    ExecuteTest(GetParam());
  }

  const auto kAllPerfTasks =
      ppc::util::MakeAllPerfTasks<InType, VlasovaAMatrixMultiplyMPI, VlasovaAMatrixMultiplySEQ>(
          PPC_SETTINGS_vlasova_a_matrix_multiply_ccs);

  const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

  const auto kPerfTestName = VlasovaMatrixMultiplyPerfTest::CustomPerfTestName;

  INSTANTIATE_TEST_SUITE_P(MatrixMultiplyPerfTests, 
                          VlasovaMatrixMultiplyPerfTest, 
                          kGtestValues, 
                          kPerfTestName);

}  // namespace vlasova_a_matrix_multiply_ccs