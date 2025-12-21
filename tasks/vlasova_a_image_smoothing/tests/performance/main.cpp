#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "vlasova_a_image_smoothing/common/include/common.hpp"
#include "vlasova_a_image_smoothing/mpi/include/ops_mpi.hpp"
#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaAImageSmoothingPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    const int width = 1024;
    const int height = 1024;
    
    input_data_.width = width;
    input_data_.height = height;
    input_data_.data.resize(static_cast<size_t>(width) * height);
    
    for (size_t i = 0; i < input_data_.data.size(); ++i) {
      input_data_.data[i] = static_cast<uint8_t>((i * 37) % 256);
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return output_data.width == input_data_.width && output_data.height == input_data_.height && !output_data.data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(VlasovaAImageSmoothingPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, VlasovaAImageSmoothingMPI, VlasovaAImageSmoothingSEQ>(
      PPC_SETTINGS_vlasova_a_image_smoothing
    );

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = VlasovaAImageSmoothingPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(PerformanceTests, VlasovaAImageSmoothingPerfTests, kGtestValues, kPerfTestName);

}  // namespace vlasova_a_image_smoothing