#include <gtest/gtest.h>

#include <cstddef>
#include <vector>

#include "util/include/perf_test_util.hpp"
#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"
#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"

namespace vlasova_a_elem_matrix_sum {

class VlasovaAElemMatrixSumPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 500;
  InType input_data_;
  OutType expected_result_;

  void SetUp() override {
    input_data_.resize(kMatrixSize_, std::vector<int>(kMatrixSize_, 1));
    expected_result_.resize(kMatrixSize_, kMatrixSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.size() != expected_result_.size()) {
      return false;
    }
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_result_[i]) {
        return false;
      }
    }
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(VlasovaAElemMatrixSumPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, VlasovaAElemMatrixSumMPI, VlasovaAElemMatrixSumSEQ>(
    PPC_SETTINGS_vlasova_a_elem_matrix_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VlasovaAElemMatrixSumPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, VlasovaAElemMatrixSumPerfTests, kGtestValues, kPerfTestName);

}  // namespace vlasova_a_elem_matrix_sum
