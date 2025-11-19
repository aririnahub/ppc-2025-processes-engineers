#include <gtest/gtest.h>
#include <vector>

#include "vlasova_a_elem_matrix_sum/common/include/common.hpp"
#include "vlasova_a_elem_matrix_sum/mpi/include/ops_mpi.hpp"
#include "vlasova_a_elem_matrix_sum/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace vlasova_a_elem_matrix_sum {

class ExampleRunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kMatrixSize_ = 1000;
  InType input_data_{};
  OutType expected_result_{};

  void SetUp() override {
    input_data_.resize(kMatrixSize_, std::vector<int>(kMatrixSize_, 1));
    expected_result_.resize(kMatrixSize_, kMatrixSize_);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data == expected_result_;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(ExampleRunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VlasovaAElemMatrixSumMPI, VlasovaAElemMatrixSumSEQ>(PPC_SETTINGS_vlasova_a_elem_matrix_sum);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = ExampleRunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, ExampleRunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace vlasova_a_elem_matrix_sum