#pragma once

#include "vlasova_a_image_smoothing/common/include/common.hpp"
#include "task/include/task.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaAImageSmoothingSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit VlasovaAImageSmoothingSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> input_image_;
  std::vector<uint8_t> output_image_;
  int window_size_ = 3;  
};

}  // namespace vlasova_a_image_smoothing