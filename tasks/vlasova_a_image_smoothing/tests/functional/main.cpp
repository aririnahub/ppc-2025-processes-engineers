#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>  
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vlasova_a_image_smoothing/common/include/common.hpp"
#include "vlasova_a_image_smoothing/mpi/include/ops_mpi.hpp"
#include "vlasova_a_image_smoothing/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaAImageSmoothingFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    window_size_ = std::get<0>(params);
   
    const int width = 64;
    const int height = 64;
    
    input_data_.width = width;
    input_data_.height = height;
    input_data_.data.resize(static_cast<size_t>(width) * height);
    
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        size_t idx = static_cast<size_t>(y) * width + x;
        // Градиент + синус для текстур
        float fx = static_cast<float>(x) / width;
        float fy = static_cast<float>(y) / height;
        float value = 128 + 64 * std::sin(fx * 10) + 64 * std::sin(fy * 8);
        input_data_.data[idx] = static_cast<uint8_t>(std::clamp(value, 0.0f, 255.0f));
      }
    }
    
    for (int i = 0; i < width * height / 20; ++i) {
      int idx = (i * 97) % input_data_.data.size();
      input_data_.data[idx] = (i % 3 == 0) ? 0 : ((i % 3 == 1) ? 255 : 128);
      }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (output_data.width != input_data_.width ||
        output_data.height != input_data_.height ||
        output_data.data.size() != input_data_.data.size()) {
      return false;
    }
    
    for (const auto& pixel : output_data.data) {
      if (pixel > 255) {
        return false;
      }
    }
    
    bool all_same = true;
    uint8_t first = output_data.data[0];
    for (size_t i = 1; i < output_data.data.size(); ++i) {
      if (output_data.data[i] != first) {
        all_same = false;
        break;
      }
    }
    
    if (all_same) {
      return false; 
    }
    
    return true;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  int window_size_ = 3;
  InType input_data_;
};

namespace {

TEST_P(VlasovaAImageSmoothingFuncTests, MedianFilterTest) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {
  std::make_tuple(3, "window3"),
  std::make_tuple(5, "window5"), 
  std::make_tuple(7, "window7")
};

const auto kTestTasksList =
    std::tuple_cat(
      ppc::util::AddFuncTask<VlasovaAImageSmoothingMPI, InType>(kTestParam, PPC_SETTINGS_vlasova_a_image_smoothing),
      ppc::util::AddFuncTask<VlasovaAImageSmoothingSEQ, InType>(kTestParam, PPC_SETTINGS_vlasova_a_image_smoothing)
    );

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = VlasovaAImageSmoothingFuncTests::PrintFuncTestName<VlasovaAImageSmoothingFuncTests>;

INSTANTIATE_TEST_SUITE_P(MedianFilterTests, VlasovaAImageSmoothingFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace vlasova_a_image_smoothing