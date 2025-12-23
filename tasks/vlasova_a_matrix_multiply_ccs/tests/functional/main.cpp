#include <gtest/gtest.h>
#include <cstdlib>
#include <ctime>
#include <tuple>
#include "vlasova_a_matrix_multiply_ccs/common/include/common.hpp"
#include "vlasova_a_matrix_multiply_ccs/mpi/include/ops_mpi.hpp"
#include "vlasova_a_matrix_multiply_ccs/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

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

  class VlasovaMatrixMultiplyFuncTest : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
  public:
    static std::string PrintTestParam(const TestType &test_param) {
      return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
    }

  protected:
    void SetUp() override {
      TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
      std::string matrix_type = std::get<1>(params);  
      
      // Инициализация генератора случайных чисел
      std::srand(static_cast<unsigned int>(std::time(nullptr)));
      
      if (matrix_type == "small") {
        A_ = generateRandomSparseMatrix(10, 10, 0.3);
        B_ = generateRandomSparseMatrix(10, 10, 0.3);
      } else if (matrix_type == "medium") {
        A_ = generateRandomSparseMatrix(50, 50, 0.1);
        B_ = generateRandomSparseMatrix(50, 50, 0.1);
      } else { 
        A_ = generateRandomSparseMatrix(100, 100, 0.05);
        B_ = generateRandomSparseMatrix(100, 100, 0.05);
      }
      
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

  namespace {

  TEST_P(VlasovaMatrixMultiplyFuncTest, MatrixMultiplyCorrectness) {
    ExecuteTest(GetParam());
  }


  const std::array<TestType, 3> kTestParams = {
      std::make_tuple(1, "small"),
      std::make_tuple(2, "medium"),
      std::make_tuple(3, "large")
  };

  const auto kTestTasksList =
      std::tuple_cat(
          ppc::util::AddFuncTask<VlasovaAMatrixMultiplyMPI, InType>(kTestParams, PPC_SETTINGS_vlasova_a_matrix_multiply_ccs),
          ppc::util::AddFuncTask<VlasovaAMatrixMultiplySEQ, InType>(kTestParams, PPC_SETTINGS_vlasova_a_matrix_multiply_ccs)
      );

  const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

  const auto kPerfTestName = VlasovaMatrixMultiplyFuncTest::PrintFuncTestName<VlasovaMatrixMultiplyFuncTest>;

  INSTANTIATE_TEST_SUITE_P(MatrixMultiplyFuncTests, 
                          VlasovaMatrixMultiplyFuncTest, 
                          kGtestValues, 
                          kPerfTestName);

  }  // namespace

}  // namespace vlasova_a_matrix_multiply_ccs