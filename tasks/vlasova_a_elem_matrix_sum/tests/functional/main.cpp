#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"
#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace vlasova_a_elem_matrix_sum{

class VlasovaARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    auto test_params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    std::string matrix_name = std::get<1>(test_params);

    std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_vlasova_a_elem_matrix_sum, "data/" + matrix_name + ".txt");
    std::ifstream file(abs_path);
    if (!file.is_open()){
      throw std::runtime_error("Failed to load matrix file: " + abs_path);
    }

    int rows, cols;
    file >> rows >> cols;

    input_data_.resize(rows, std::vector<int>(cols));
    for (int i = 0; i < rows; ++i){
      for (int j = 0; j < cols; ++j){
        file >> input_data_[i][j];
      }
    }

    expected_result_.resize(rows);
    for (int i = 0; i < rows; ++i){
      int sum = 0;
      for (int j = 0; j < cols; ++j){
        sum += input_data_[i][j];
      }
      expected_result_[i] = sum;
    }
  }
 
  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_result_.size()){
      return false;
    } 
    for (int i = 0; i<output_data.size(); ++i){
      if (output_data[i]!= expected_result_[i]){
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_result_;
};

namespace {

TEST_P(VlasovaARunFuncTestsProcesses, MatrixRowSum) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "matrix1"), std::make_tuple(5, "matrix2"), std::make_tuple(7, "matrix3")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<VlasovaAElemMatrixSumMPI, InType>(kTestParam, PPC_SETTINGS_vlasova_a_elem_matrix_sum),
                   ppc::util::AddFuncTask<VlasovaAElemMatrixSumSEQ, InType>(kTestParam, PPC_SETTINGS_vlasova_a_elem_matrix_sum));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VlasovaARunFuncTestsProcesses::PrintFuncTestName<VlasovaARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(MatrixRowSum, VlasovaARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vlasova_a_elem_matrix_sum
