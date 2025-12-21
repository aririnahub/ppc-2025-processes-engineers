#pragma once

#include "vlasova_a_image_smoothing/common/include/common.hpp"

namespace vlasova_a_image_smoothing {

class VlasovaAImageSmoothingMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit VlasovaAImageSmoothingMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
  
  uint8_t GetPixelMedian(int x, int y, int overlap_start, const std::vector<uint8_t>& local_data) const;
  
  int width_ = 0;
  int height_ = 0;
  int window_size_ = 3;
};

}  // namespace vlasova_a_image_smoothing